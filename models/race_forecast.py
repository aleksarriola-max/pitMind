"""Battle prediction model — projects positions forward from a given lap."""

import numpy as np
import pandas as pd
from models.constants import PIT_LANE_DELTA, DEFAULT_PIT_LANE_DELTA, SC_PIT_LANE_DELTA

# Module-level cache for quadratic degradation coefficients, populated by
# get_compound_degradation_rates() and consumed by compare_pit_scenarios().
_COMPOUND_POLY: dict = {}


def forecast_positions(df: pd.DataFrame, from_lap: int, n_laps: int = 15) -> tuple[pd.DataFrame, list[dict]]:
    """
    Project positions for each driver n_laps ahead starting from from_lap.

    Uses gap_ahead and pace_delta to estimate when overtakes will occur.

    Returns:
        proj_df: DataFrame with columns [lap, driver, projected_position, projected_gap_ahead]
        aggression_windows: list of {lap, driver_behind, driver_ahead, projected_gap, is_drs_range}
    """
    snap = df[df["lap"] == from_lap].copy()
    if len(snap) == 0:
        # Fall back to nearest lap
        available = df["lap"].unique()
        from_lap = available[np.argmin(np.abs(available - from_lap))]
        snap = df[df["lap"] == from_lap].copy()

    snap = snap.sort_values("position").reset_index(drop=True)
    drivers = snap["driver"].tolist()

    # Build state arrays: position, gap_ahead (to car ahead), pace_delta
    positions = {row["driver"]: int(row["position"]) if pd.notna(row["position"]) else i + 1
                 for i, (_, row) in enumerate(snap.iterrows())}
    gaps = {row["driver"]: float(row["gap_ahead"]) if pd.notna(row.get("gap_ahead")) and row.get("gap_ahead") != 0 else 2.0
            for _, row in snap.iterrows()}
    pace = {row["driver"]: float(row["pace_delta"]) if pd.notna(row.get("pace_delta")) else 0.0
            for _, row in snap.iterrows()}

    # Sort drivers by current position
    ordered = sorted(drivers, key=lambda d: positions.get(d, 20))

    records = []
    aggression_windows = []

    # State: simulated gap for each consecutive pair
    sim_gaps = {}
    for i in range(len(ordered) - 1):
        ahead = ordered[i]
        behind = ordered[i + 1]
        sim_gaps[(behind, ahead)] = gaps.get(behind, 2.0)

    current_order = list(ordered)

    for lap_offset in range(n_laps + 1):
        proj_lap = from_lap + lap_offset

        # Update gaps based on pace difference
        new_gaps = {}
        for i in range(len(current_order) - 1):
            ahead_d = current_order[i]
            behind_d = current_order[i + 1]
            pair = (behind_d, ahead_d)

            current_gap = sim_gaps.get(pair, 2.0)
            # Positive pace_delta means slower — so faster driver has lower pace_delta
            pace_advantage = pace.get(behind_d, 0) - pace.get(ahead_d, 0)
            # pace_advantage > 0 means driver behind is SLOWER (bigger delta = slower)
            # pace_advantage < 0 means driver behind is FASTER (closing)
            closing_rate = -pace_advantage  # positive = closing
            new_gap = max(0.0, current_gap - closing_rate)
            new_gaps[pair] = new_gap

        sim_gaps = new_gaps

        # Detect and simulate overtakes
        swap_occurred = True
        while swap_occurred:
            swap_occurred = False
            for i in range(len(current_order) - 1):
                ahead_d = current_order[i]
                behind_d = current_order[i + 1]
                pair = (behind_d, ahead_d)
                gap = sim_gaps.get(pair, 2.0)

                if gap <= 0:
                    # Overtake: swap positions
                    current_order[i], current_order[i + 1] = current_order[i + 1], current_order[i]
                    sim_gaps[pair] = 0.5  # post-overtake gap
                    swap_occurred = True
                    break

        # Record state
        for pos_i, driver in enumerate(current_order):
            ahead_d = current_order[pos_i - 1] if pos_i > 0 else None
            gap_val = sim_gaps.get((driver, ahead_d), np.nan) if ahead_d else np.nan
            records.append({
                "lap": proj_lap,
                "driver": driver,
                "projected_position": pos_i + 1,
                "projected_gap_ahead": gap_val,
            })

            # Flag aggression windows
            if ahead_d and pd.notna(gap_val) and gap_val <= 2.0 and lap_offset > 0:
                aggression_windows.append({
                    "lap": proj_lap,
                    "driver_behind": driver,
                    "driver_ahead": ahead_d,
                    "projected_gap": round(gap_val, 3),
                    "is_drs_range": gap_val <= 1.0,
                })

    proj_df = pd.DataFrame(records)
    return proj_df, aggression_windows


def get_aggression_zone_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute position-change frequency by race third (early/mid/late).
    Returns DataFrame: driver x {early_changes, mid_changes, late_changes}
    """
    if "position" not in df.columns or len(df) == 0:
        return pd.DataFrame()

    max_lap = df["lap"].max()
    early_end = max_lap // 3
    mid_end = 2 * max_lap // 3

    def zone(lap):
        if lap <= early_end:
            return "Early"
        elif lap <= mid_end:
            return "Mid"
        return "Late"

    df2 = df.sort_values(["driver", "lap"]).copy()
    df2["zone"] = df2["lap"].apply(zone)
    df2["pos_change"] = df2.groupby("driver")["position"].diff().abs().fillna(0)

    result = df2.groupby(["driver", "zone"])["pos_change"].sum().unstack(fill_value=0)
    for col in ["Early", "Mid", "Late"]:
        if col not in result.columns:
            result[col] = 0
    return result[["Early", "Mid", "Late"]].reset_index()


def get_compound_degradation_rates(df: pd.DataFrame) -> dict:
    """Per-compound degradation rate (s/lap) via quadratic regression.

    Fits a degree-2 polynomial (a*t² + b*t + c) per compound to capture the
    three tyre phases: warm-up (laps 1-3, pace improves), stable decline, and
    cliff (late-stint acceleration). Returns the slope at lap 15 of a fresh
    stint as the representative scalar rate, floored at 0.005 s/lap.

    Also populates _COMPOUND_POLY with (a, b, c) coefficients for use in
    compare_pit_scenarios() honeymoon simulation.
    """
    rates = {}
    for compound, group in df.groupby("tyre_compound"):
        clean = group[["tyre_age", "pace_delta"]].dropna()
        if len(clean) < 10:
            continue
        coeffs = np.polyfit(clean["tyre_age"].values, clean["pace_delta"].values, 2)
        a, b = float(coeffs[0]), float(coeffs[1])
        # d/dt(a*t² + b*t + c) at t=15 → representative mid-stint slope
        slope_at_15 = 2 * a * 15 + b
        rates[str(compound)] = float(max(slope_at_15, 0.005))
        _COMPOUND_POLY[str(compound)] = (a, b, float(coeffs[2]))
    return rates


def compare_pit_scenarios(
    df: pd.DataFrame,
    race_slug: str,
    from_lap: int,
    driver: str,
    n_laps: int = 15,
) -> list[dict]:
    """Run 4 pit timing scenarios forward from from_lap and compare outcomes.

    Scenarios: Pit Now (offset 0), Pit +3, Pit +5, Stay Out.

    For each pit scenario:
      - At the pit lap, the driver's gap_ahead increases by the circuit's pit lane
        delta (they re-enter behind their pre-pit position).
      - tyre_age resets to 0 and pace_delta resets to 0 for the pit lap and the
        next 5 laps (fresh-tyre honeymoon period).
      - forecast_positions() is then run on this modified DataFrame.

    Returns a list of 4 dicts:
        label              str   — "Pit Now" / "Pit +3" / "Pit +5" / "Stay Out"
        pit_lap            int|None — actual lap number of the pit stop (None for Stay Out)
        projected_position int   — projected P at from_lap + n_laps
        projected_gap_ahead float|None
        position_delta     int   — current_position - projected_position (positive = gained)
        recommendation     bool  — True for the single best scenario

    Returns [] if driver not in df["driver"].
    """
    if driver not in df["driver"].unique():
        return []

    pit_delta_s = PIT_LANE_DELTA.get(race_slug, DEFAULT_PIT_LANE_DELTA)

    # Snap from_lap to nearest available lap
    snap = df[df["lap"] == from_lap]
    if len(snap) == 0:
        available = df["lap"].unique()
        from_lap = int(available[np.argmin(np.abs(available - from_lap))])
        snap = df[df["lap"] == from_lap]

    driver_snap = snap[snap["driver"] == driver]
    if len(driver_snap) == 0:
        return []

    current_pos_val = driver_snap["position"].iloc[0]
    current_position = int(current_pos_val) if pd.notna(current_pos_val) else 10

    # (label, pit_offset)  — None offset means no pit stop
    scenario_defs = [
        ("Pit Now",  0),
        ("Pit +3",   3),
        ("Pit +5",   5),
        ("Stay Out", None),
    ]

    results = []

    _deg_rates = get_compound_degradation_rates(df)
    _tyre_cmp = str(driver_snap["tyre_compound"].iloc[0]) \
                if "tyre_compound" in driver_snap.columns \
                   and pd.notna(driver_snap["tyre_compound"].iloc[0]) \
                else "MEDIUM"
    _rate = _deg_rates.get(_tyre_cmp, 0.02)

    for label, offset in scenario_defs:
        try:
            scenario_df = df.copy()

            if offset is not None:
                pit_lap = from_lap + offset
                # Reduce pit cost under Safety Car
                _effective_pit_delta = pit_delta_s
                _sc_snap = scenario_df[
                    (scenario_df["driver"] == driver) & (scenario_df["lap"] == pit_lap)
                ]
                if ("safety_car_active" in scenario_df.columns
                        and len(_sc_snap) > 0
                        and bool(_sc_snap["safety_car_active"].iloc[0])):
                    _effective_pit_delta = SC_PIT_LANE_DELTA
                pit_mask = (scenario_df["driver"] == driver) & (scenario_df["lap"] == pit_lap)

                if pit_mask.any():
                    existing_gap = scenario_df.loc[pit_mask, "gap_ahead"].values[0]
                    if pd.isna(existing_gap):
                        existing_gap = 2.0
                    _pre_pit_pace_val = scenario_df.loc[pit_mask, "pace_delta"].values[0]
                    _pre_pit_pace = float(_pre_pit_pace_val) if pd.notna(_pre_pit_pace_val) else 0.0
                    _fresh_pace = min(_pre_pit_pace, 0.0)
                    # Re-entry: driver loses pit_lane_delta seconds to the car ahead
                    scenario_df.loc[pit_mask, "gap_ahead"] = float(existing_gap) + _effective_pit_delta
                    scenario_df.loc[pit_mask, "tyre_age"] = 0
                    scenario_df.loc[pit_mask, "pace_delta"] = _fresh_pace

                    # Fresh-tyre honeymoon: tyre_age reset for next 5 laps
                    post_pit_mask = (
                        (scenario_df["driver"] == driver)
                        & (scenario_df["lap"] > pit_lap)
                        & (scenario_df["lap"] <= pit_lap + 5)
                    )
                    if post_pit_mask.any():
                        post_laps = scenario_df.loc[post_pit_mask, "lap"]
                        scenario_df.loc[post_pit_mask, "tyre_age"] = (post_laps - pit_lap).values
                    # Per-compound degradation curve for pace_delta
                    for _hl in range(1, 6):
                        _hl_mask = (
                            (scenario_df["driver"] == driver)
                            & (scenario_df["lap"] == pit_lap + _hl)
                        )
                        if _hl_mask.any():
                            if str(_tyre_cmp) in _COMPOUND_POLY:
                                _a, _b, _c = _COMPOUND_POLY[str(_tyre_cmp)]
                                _proj_pace = _fresh_pace + (_a * _hl ** 2 + _b * _hl)
                            else:
                                _proj_pace = _fresh_pace + _rate * _hl
                            scenario_df.loc[_hl_mask, "pace_delta"] = _proj_pace
            else:
                pit_lap = None

            proj_df, _ = forecast_positions(scenario_df, from_lap=from_lap, n_laps=n_laps)

            end_lap = from_lap + n_laps
            driver_proj = proj_df[
                (proj_df["driver"] == driver) & (proj_df["lap"] == end_lap)
            ]

            if len(driver_proj) == 0:
                # Fewer laps available — take last projected row for this driver
                driver_proj = proj_df[proj_df["driver"] == driver].sort_values("lap").tail(1)

            if len(driver_proj) == 0:
                projected_position = current_position
                projected_gap_ahead = None
            else:
                proj_row = driver_proj.iloc[0]
                projected_position = int(proj_row["projected_position"])
                gap_val = proj_row.get("projected_gap_ahead", np.nan)
                projected_gap_ahead = (
                    round(float(gap_val), 3) if pd.notna(gap_val) else None
                )

            results.append({
                "label":               label,
                "pit_lap":             int(pit_lap) if pit_lap is not None else None,
                "projected_position":  projected_position,
                "projected_gap_ahead": projected_gap_ahead,
                "position_delta":      current_position - projected_position,
                "recommendation":      False,
            })
            # Attach trajectory for UI visualization (prefixed keys, not in schema)
            _drv_traj = proj_df[proj_df["driver"] == driver].sort_values("lap")
            results[-1]["_traj_laps"] = _drv_traj["lap"].tolist()
            results[-1]["_traj_positions"] = _drv_traj["projected_position"].tolist()

        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning(
                f"compare_pit_scenarios: scenario '{label}' failed: {e}"
            )
            results.append({
                "label":               label,
                "pit_lap":             int(from_lap + offset) if offset is not None else None,
                "projected_position":  current_position,
                "projected_gap_ahead": None,
                "position_delta":      0,
                "recommendation":      False,
            })

    # Best: most positions gained; tiebreak = latest pit lap (preserve track position longer)
    if results:
        best = max(
            results,
            key=lambda s: (s["position_delta"], -(s["pit_lap"] if s["pit_lap"] is not None else 0))
        )
        best["recommendation"] = True

    return results

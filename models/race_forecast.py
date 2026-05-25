"""Battle prediction model — projects positions forward from a given lap."""

import numpy as np
import pandas as pd


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

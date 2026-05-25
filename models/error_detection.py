"""Pilot error and strategy oversight detection.

Detects 5 error types from per-lap data:
  LOCKUP        — lap time spike not explained by flags
  PACE_COLLAPSE — sustained degradation faster than tyre age model
  LATE_PIT      — pitted after pit_window_pressure was critical for 5+ laps
  SC_LOSS       — position lost on first green lap after a safety car
  RADIO_STRESS  — negative radio sentiment with no on-track event
"""

import numpy as np
import pandas as pd


def detect_pilot_errors(df: pd.DataFrame, flag_periods: list | None = None) -> pd.DataFrame:
    """
    Detect pilot errors for all drivers in df.

    Returns DataFrame: lap, driver, error_type, severity, signal_value, description
    """
    flag_laps = _flag_lap_set(flag_periods or [])
    sc_periods = [p for p in (flag_periods or []) if p["flag"] == "SAFETY_CAR"]

    errors = []
    for driver, grp in df.groupby("driver"):
        grp = grp.sort_values("lap").reset_index(drop=True)
        errors.extend(_detect_lockups(grp, driver, flag_laps))
        errors.extend(_detect_pace_collapse(grp, driver, flag_laps))
        errors.extend(_detect_late_pit(grp, driver))
        errors.extend(_detect_sc_loss(grp, driver, sc_periods))
        errors.extend(_detect_radio_stress(grp, driver, flag_laps))

    if not errors:
        return pd.DataFrame(columns=["lap", "driver", "error_type", "severity", "signal_value", "description"])

    return pd.DataFrame(errors).sort_values(["driver", "lap"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flag_lap_set(flag_periods: list) -> set:
    laps = set()
    for p in flag_periods:
        laps.update(range(p["lap_start"], p["lap_end"] + 1))
    return laps


def _detect_lockups(grp: pd.DataFrame, driver: str, flag_laps: set) -> list:
    """Lap time spike >1.5s above personal stint best, outside flag periods."""
    errors = []
    if "lap_time" not in grp.columns or "stint" not in grp.columns:
        return errors

    for stint_id, stint_grp in grp.groupby("stint"):
        valid = stint_grp[stint_grp["lap_time"].notna()]
        if len(valid) < 3:
            continue
        best = valid["lap_time"].quantile(0.15)  # 15th pct as "normal" pace
        for _, row in valid.iterrows():
            lap = int(row["lap"])
            if lap in flag_laps:
                continue
            delta = row["lap_time"] - best
            if delta > 1.5:
                severity = "major" if delta > 3.0 else "minor"
                errors.append({
                    "lap": lap, "driver": driver,
                    "error_type": "LOCKUP",
                    "severity": severity,
                    "signal_value": round(delta, 2),
                    "description": f"Lap time {delta:.1f}s slower than stint pace — possible lock-up or off-track",
                })
    return errors


def _detect_pace_collapse(grp: pd.DataFrame, driver: str, flag_laps: set) -> list:
    """3+ consecutive laps with pace_delta worsening >0.3s/lap vs. tyre age trend."""
    errors = []
    if "pace_delta" not in grp.columns or grp["pace_delta"].isna().all():
        return errors

    clean = grp[~grp["lap"].isin(flag_laps) & grp["pace_delta"].notna()].copy()
    if len(clean) < 5:
        return errors

    clean["pd_diff"] = clean["pace_delta"].diff()
    window = 3
    consecutive = 0
    collapse_start = None

    for i, row in clean.iterrows():
        if pd.notna(row["pd_diff"]) and row["pd_diff"] > 0.3:
            consecutive += 1
            if consecutive == 1:
                collapse_start = int(row["lap"])
            if consecutive >= window:
                errors.append({
                    "lap": collapse_start, "driver": driver,
                    "error_type": "PACE_COLLAPSE",
                    "severity": "major",
                    "signal_value": round(clean.loc[i, "pd_diff"], 2),
                    "description": f"{window}+ consecutive laps of accelerating pace loss from lap {collapse_start}",
                })
                consecutive = 0  # reset to avoid duplicate
        else:
            consecutive = 0

    return errors


def _detect_late_pit(grp: pd.DataFrame, driver: str) -> list:
    """Pitted after pit_window_pressure ≥ 80 for 5+ consecutive laps."""
    errors = []
    if "is_pit_in" not in grp.columns or "pit_window_pressure" not in grp.columns:
        return errors

    pit_laps = grp[grp["is_pit_in"]]["lap"].tolist()
    for pit_lap in pit_laps:
        before = grp[grp["lap"] < pit_lap].tail(8)
        if len(before) < 5:
            continue
        high_pressure = (before["pit_window_pressure"] >= 80).sum()
        if high_pressure >= 5:
            avg_pressure = round(float(before["pit_window_pressure"].mean()), 1)
            errors.append({
                "lap": int(pit_lap), "driver": driver,
                "error_type": "LATE_PIT",
                "severity": "minor",
                "signal_value": avg_pressure,
                "description": f"Pit window pressure at {avg_pressure}/100 for {high_pressure} laps before pitting on lap {int(pit_lap)}",
            })
    return errors


def _detect_sc_loss(grp: pd.DataFrame, driver: str, sc_periods: list) -> list:
    """Position lost on the first green lap after each safety car."""
    errors = []
    if "position" not in grp.columns or not sc_periods:
        return errors

    for period in sc_periods:
        first_green = period["lap_end"] + 1
        sc_lap_row = grp[grp["lap"] == period["lap_end"]]
        green_lap_row = grp[grp["lap"] == first_green]
        if len(sc_lap_row) == 0 or len(green_lap_row) == 0:
            continue
        pos_during_sc = sc_lap_row.iloc[0]["position"]
        pos_after = green_lap_row.iloc[0]["position"]
        if pd.notna(pos_during_sc) and pd.notna(pos_after):
            lost = int(pos_after) - int(pos_during_sc)
            if lost > 0:
                errors.append({
                    "lap": first_green, "driver": driver,
                    "error_type": "SC_LOSS",
                    "severity": "major" if lost > 1 else "minor",
                    "signal_value": float(lost),
                    "description": f"Lost {lost} position(s) on SC restart lap {first_green}",
                })
    return errors


def _detect_radio_stress(grp: pd.DataFrame, driver: str, flag_laps: set) -> list:
    """High radio stress on laps with no flag/SC event."""
    errors = []
    if "radio_sentiment" not in grp.columns or grp["radio_sentiment"].isna().all():
        return errors

    stress_rows = grp[
        grp["radio_sentiment"].notna() &
        (grp["radio_sentiment"] < -0.7) &
        (~grp["lap"].isin(flag_laps))
    ]
    for _, row in stress_rows.iterrows():
        errors.append({
            "lap": int(row["lap"]), "driver": driver,
            "error_type": "RADIO_STRESS",
            "severity": "minor",
            "signal_value": round(float(row["radio_sentiment"]), 2),
            "description": f"High radio stress (sentiment {row['radio_sentiment']:.2f}) on lap {int(row['lap'])} — no on-track event",
        })
    return errors


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def error_counts(error_df: pd.DataFrame) -> dict:
    """Return {driver: {error_type: count}} summary."""
    if len(error_df) == 0:
        return {}
    result = {}
    for driver, grp in error_df.groupby("driver"):
        result[driver] = grp["error_type"].value_counts().to_dict()
    return result


def total_errors(error_df: pd.DataFrame, driver: str) -> int:
    return len(error_df[error_df["driver"] == driver]) if len(error_df) > 0 else 0

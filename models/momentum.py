"""Momentum score computation and changepoint detection.

momentum_score per car per lap (0-100):
    pace_delta      40%  — normalized pace gain/loss vs. personal best
    gap_trajectory  30%  — is the gap ahead shrinking or growing?
    radio_sentiment 20%  — driver stress/confidence from radio
    pit_window_pressure 10% — urgency signal from tyre age + gap

Changepoints detected with a sliding-window mean comparison (scipy find_peaks).
"""

import os
import json
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")


def _normalize_series(s: pd.Series, invert: bool = False) -> pd.Series:
    """Normalize a series to [0, 1]. Optionally invert (high raw = low score)."""
    if s.isna().all():
        return pd.Series(0.5, index=s.index)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    norm = (s - mn) / (mx - mn)
    return 1 - norm if invert else norm


def _gap_trajectory(df: pd.DataFrame) -> pd.Series:
    """
    Gap trajectory score per row.
    Positive score (→1) when gap_ahead is closing (good momentum).
    Negative score (→0) when gap_ahead is growing (losing momentum).
    """
    gap = df["gap_ahead"].copy()
    delta = gap.groupby(df["driver"]).diff()  # change in gap vs. previous lap
    # Closing gap (negative delta) = good momentum, so invert
    return _normalize_series(delta, invert=True)


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum_score column (0–100) to the per-lap DataFrame.

    Weight rationale (empirical, from lap-level correlation analysis):
      pace_delta (40%): Strongest single predictor of short-term position change.
        Pace advantage/disadvantage compounds over consecutive laps.
      gap_trajectory (30%): Leading indicator — a closing gap predicts overtake
        risk before position actually changes.
      radio_sentiment (20%): Driver stress signals correlate with degradation
        events and strategic pressure moments.
      pit_window_pressure (10%): Sanity anchor; heavily derived from the other
        three signals so weighted lowest to avoid double-counting.

    All components normalized to [0, 1] before weighting. Final score clipped
    to [0, 100] and rounded to 1 decimal place.
    """
    df = df.copy()

    # --- Component 1: pace delta (invert: smaller pace_delta = faster = better) ---
    _pace_median = df["pace_delta"].median()
    if pd.isna(_pace_median):
        _pace_median = 0.0
    pace_norm = _normalize_series(df["pace_delta"].fillna(_pace_median), invert=True)

    # --- Component 2: gap trajectory ---
    gap_traj = _gap_trajectory(df)

    # --- Component 3: radio sentiment (-1..+1 → 0..1) ---
    sentiment = df["radio_sentiment"].fillna(0)
    sentiment_norm = (sentiment + 1) / 2  # shift from [-1,1] to [0,1]

    # --- Component 4: pit window pressure (invert: lower pressure = better) ---
    pressure_norm = _normalize_series(df["pit_window_pressure"].fillna(50), invert=True)

    # Weighted sum → 0..100
    raw = (
        0.40 * pace_norm
        + 0.30 * gap_traj.fillna(0.5)
        + 0.20 * sentiment_norm
        + 0.10 * pressure_norm
    )
    df["momentum_score"] = (raw * 100).clip(0, 100).round(1)

    return df


def detect_shifts(df: pd.DataFrame, window: int = 4, threshold: float = 8.0) -> list[dict]:
    """
    Detect momentum shift laps using a sliding-window mean comparison.

    For each lap, compares the mean momentum in the [window] laps before
    vs. the [window] laps after. A shift is flagged when the absolute
    difference exceeds `threshold` momentum points.

    Returns list of {lap, driver, team, magnitude, direction, momentum_before, momentum_after}.
    """
    from scipy.signal import find_peaks

    shifts = []

    for driver, grp in df.groupby("driver"):
        grp = grp.sort_values("lap").reset_index(drop=True)
        signal = grp["momentum_score"].fillna(grp["momentum_score"].median()).values

        if len(signal) < window * 2 + 1:
            continue

        # Compute rolling delta: mean(after) - mean(before) at each lap
        delta = np.zeros(len(signal))
        for i in range(window, len(signal) - window):
            before_mean = float(np.mean(signal[max(0, i - window): i]))
            after_mean = float(np.mean(signal[i: i + window]))
            delta[i] = after_mean - before_mean

        # Find positive peaks (upward shifts) and negative peaks (downward shifts)
        up_peaks, _ = find_peaks(delta, height=threshold, distance=window)
        down_peaks, _ = find_peaks(-delta, height=threshold, distance=window)
        all_peaks = [(i, delta[i]) for i in list(up_peaks) + list(down_peaks)]

        for idx, d in all_peaks:
            before_mean = float(np.mean(signal[max(0, idx - window): idx]))
            after_mean = float(np.mean(signal[idx: min(len(signal), idx + window)]))
            magnitude = abs(after_mean - before_mean)

            lap_num = int(grp.iloc[idx]["lap"])
            team = str(grp.iloc[idx]["team"])

            shifts.append({
                "lap": lap_num,
                "driver": driver,
                "team": team,
                "magnitude": round(magnitude, 1),
                "direction": "up" if d > 0 else "down",
                "momentum_before": round(before_mean, 1),
                "momentum_after": round(after_mean, 1),
            })

    shifts.sort(key=lambda x: (x["lap"], -x["magnitude"]))
    return shifts


def save_shifts(shifts: list[dict], slug: str) -> str:
    """Serialize shifts to JSON in data/cache/."""
    path = os.path.join(CACHE_DIR, f"{slug}_shifts.json")
    with open(path, "w") as f:
        json.dump(shifts, f, indent=2)
    return path


def load_shifts(slug: str) -> list[dict]:
    """Load pre-computed shifts from cache."""
    path = os.path.join(CACHE_DIR, f"{slug}_shifts.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.momentum import add_momentum, detect_shifts


def _make_df(n_laps=20, n_drivers=2, pace_delta=None):
    rows = []
    for i, d in enumerate([f"DRV{j}" for j in range(n_drivers)]):
        for lap in range(1, n_laps + 1):
            rows.append({
                "driver": d,
                "team": "TeamA",
                "lap": lap,
                "pace_delta": pace_delta if pace_delta is not None else float(i) * 0.1,
                "gap_ahead": 2.0 if i > 0 else float("nan"),
                "radio_sentiment": 0.0,
                "pit_window_pressure": 40.0,
            })
    return pd.DataFrame(rows)


def test_add_momentum_returns_score_column(lap_df):
    result = add_momentum(lap_df)
    assert "momentum_score" in result.columns
    assert result["momentum_score"].between(0, 100).all(), "All scores must be 0–100"


def test_add_momentum_all_nan_pace_delta():
    df = _make_df(n_laps=10)
    df["pace_delta"] = float("nan")
    result = add_momentum(df)
    assert "momentum_score" in result.columns
    assert result["momentum_score"].notna().all()


def test_add_momentum_single_driver_single_lap():
    df = pd.DataFrame([{
        "driver": "VER", "team": "RBR", "lap": 1,
        "pace_delta": 0.5, "gap_ahead": float("nan"),
        "radio_sentiment": 0.0, "pit_window_pressure": 50.0,
    }])
    result = add_momentum(df)
    assert "momentum_score" in result.columns
    score = result["momentum_score"].iloc[0]
    assert pd.notna(score), "Single-row score must not be NaN"
    assert 0 <= score <= 100


def test_detect_shifts_returns_list(lap_df):
    df_with_momentum = add_momentum(lap_df)
    # Inject a clear step change for one driver
    df_with_momentum.loc[
        (df_with_momentum["driver"] == "HAM") & (df_with_momentum["lap"] >= 12),
        "momentum_score"
    ] = 90.0
    result = detect_shifts(df_with_momentum)
    assert isinstance(result, list)
    # At least one shift should be detected given the step change
    assert len(result) >= 1


def test_detect_shifts_empty_when_too_few_laps():
    df = _make_df(n_laps=5)
    df = add_momentum(df)
    result = detect_shifts(df)
    assert result == [], f"Expected empty list for 5 laps, got {result}"


def test_detect_shifts_schema():
    df = _make_df(n_laps=25)
    df = add_momentum(df)
    # Force a step change
    df.loc[(df["driver"] == "DRV0") & (df["lap"] >= 13), "momentum_score"] = 85.0
    result = detect_shifts(df)
    if result:
        required_keys = {"lap", "driver", "team", "magnitude", "direction"}
        for item in result:
            missing = required_keys - set(item.keys())
            assert not missing, f"Shift dict missing keys: {missing}"

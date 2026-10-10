import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.driver_soul import compute_incident_risk, get_lap_state, _compute_validation_metrics


# ---------------------------------------------------------------------------
# compute_incident_risk — takes DataFrame, returns Series
# ---------------------------------------------------------------------------

def test_incident_risk_returns_series_in_range():
    df = pd.DataFrame([{
        "tyre_age": 15, "aggression_level": 50, "pressure_consistency": 50,
    }])
    risk = compute_incident_risk(df)
    assert isinstance(risk, pd.Series)
    assert float(risk.iloc[0]) >= 0.0
    assert float(risk.iloc[0]) <= 1.0


def test_incident_risk_high_when_all_extreme():
    """Max tyre age, max aggression, min consistency → risk near 1."""
    df = pd.DataFrame([{
        "tyre_age": 50, "aggression_level": 100, "pressure_consistency": 0,
    }])
    risk = compute_incident_risk(df)
    assert float(risk.iloc[0]) > 0.7


def test_incident_risk_low_when_all_calm():
    """Fresh tyres, low aggression, high consistency → risk near 0."""
    df = pd.DataFrame([{
        "tyre_age": 2, "aggression_level": 10, "pressure_consistency": 90,
    }])
    risk = compute_incident_risk(df)
    assert float(risk.iloc[0]) < 0.3


def test_incident_risk_handles_nan_traits():
    """NaN inputs must not raise. aggression/consistency use fillna; tyre_age propagates NaN."""
    df = pd.DataFrame([{
        "tyre_age": np.nan, "aggression_level": np.nan, "pressure_consistency": np.nan,
    }])
    risk = compute_incident_risk(df)
    assert isinstance(risk, pd.Series)
    # tyre_age NaN propagates through clip → result is NaN (acceptable; not a crash)
    assert len(risk) == 1


def test_incident_risk_multiple_rows(lap_df):
    """Works on multi-row race DataFrame if trait columns are present."""
    df = lap_df.copy()
    df["aggression_level"] = 50
    df["pressure_consistency"] = 60
    risk = compute_incident_risk(df)
    assert len(risk) == len(df)
    assert (risk >= 0.0).all()
    assert (risk <= 1.0).all()


# ---------------------------------------------------------------------------
# get_lap_state — returns dict of trait + prediction values
# ---------------------------------------------------------------------------

def test_get_lap_state_returns_dict(lap_df):
    state = get_lap_state(lap_df, "VER", lap=5)
    assert isinstance(state, dict)


def test_get_lap_state_missing_lap_returns_empty(lap_df):
    state = get_lap_state(lap_df, "VER", lap=999)
    assert state == {}


# ---------------------------------------------------------------------------
# _compute_validation_metrics — build-time split validation
# ---------------------------------------------------------------------------

def test_validation_metrics_returns_empty_without_race_slug(lap_df):
    """No race_slug column → returns {}."""
    metrics = _compute_validation_metrics(lap_df, None, [])
    assert metrics == {}


def test_validation_metrics_returns_empty_with_single_race(lap_df):
    """Only one race → can't do train/test split → returns {}."""
    df = lap_df.copy()
    df["race_slug"] = "bahrain_2025"
    metrics = _compute_validation_metrics(df, None, [])
    assert metrics == {}


# ---------------------------------------------------------------------------
# _build_labels — labels must not look across race boundaries
# ---------------------------------------------------------------------------

def test_labels_do_not_cross_races():
    """A driver's last lap in race A must not be labeled from race B's laps."""
    from models.driver_soul import _build_labels
    rows = []
    # Race A: driver holds P5 on laps 1-3 (no change ahead)
    for lap in (1, 2, 3):
        rows.append({"race_slug": "a", "driver": "VER", "lap": lap, "position": 5, "is_pit_in": False})
    # Race B: same driver starts P1 on laps 1-3
    for lap in (1, 2, 3):
        rows.append({"race_slug": "b", "driver": "VER", "lap": lap, "position": 1, "is_pit_in": False})
    out = _build_labels(pd.DataFrame(rows))
    race_a = out[out["race_slug"] == "a"]
    # Without the fix, race A laps would 'see' race B's P1 and get gain=1
    assert race_a["_target_gain"].sum() == 0
    assert race_a["_target_loss"].sum() == 0


def test_validation_holds_out_last_race():
    """Validation must report a race that is not among the training races."""
    rng = np.random.default_rng(0)
    rows = []
    for slug in ("r1", "r2", "r3"):
        for drv in ("AAA", "BBB", "CCC", "DDD"):
            for lap in range(1, 31):
                rows.append({"race_slug": slug, "driver": drv, "lap": lap,
                             "tyre_age": lap % 20, "gap_ahead": rng.random(), "gap_behind": rng.random(),
                             "pit_window_pressure": rng.random() * 100, "lap_time": 90 + rng.random(),
                             "_target_gain": int(rng.random() > .5), "_target_loss": int(rng.random() > .5),
                             "_target_pit": int(rng.random() > .8)})
    df = pd.DataFrame(rows)
    feats = ["tyre_age", "gap_ahead", "gap_behind", "pit_window_pressure", "lap_time"]
    metrics = _compute_validation_metrics(df, predictor=None, feature_names=feats)
    assert metrics, "expected metrics"
    for m in metrics.values():
        assert m["val_race"] not in m["train_races"]
        assert "majority_baseline" in m and "roc_auc" in m

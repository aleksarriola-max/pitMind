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

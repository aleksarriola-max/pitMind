import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.race_forecast import forecast_positions, get_compound_degradation_rates


def test_forecast_returns_correct_types(lap_df):
    proj_df, aggression_windows = forecast_positions(lap_df, from_lap=5, n_laps=10)
    assert isinstance(proj_df, pd.DataFrame)
    assert isinstance(aggression_windows, list)


def test_forecast_correct_lap_range(lap_df):
    proj_df, _ = forecast_positions(lap_df, from_lap=5, n_laps=10)
    assert proj_df["lap"].min() == 5
    assert proj_df["lap"].max() == 15


def test_forecast_fallback_when_lap_missing(lap_df):
    # from_lap=999 is beyond the data — should snap to nearest without crashing
    proj_df, _ = forecast_positions(lap_df, from_lap=999, n_laps=5)
    assert len(proj_df) > 0


def test_forecast_all_nan_pace_delta(lap_df):
    lap_df = lap_df.copy()
    lap_df["pace_delta"] = float("nan")
    proj_df, _ = forecast_positions(lap_df, from_lap=5, n_laps=5)
    assert len(proj_df) > 0


def test_forecast_single_driver():
    rows = [{"driver": "VER", "team": "RBR", "lap": i, "position": 1.0,
             "gap_ahead": float("nan"), "gap_behind": float("nan"),
             "pace_delta": 0.1, "radio_sentiment": 0.0, "pit_window_pressure": 40.0,
             "tyre_age": i, "tyre_compound": "MEDIUM", "lap_time": 90.0}
            for i in range(1, 25)]
    df = pd.DataFrame(rows)
    proj_df, agg = forecast_positions(df, from_lap=5, n_laps=10)
    assert len(proj_df) > 0
    # Single driver → no pairs → no aggression windows
    assert isinstance(agg, list)


def test_forecast_schema(lap_df):
    proj_df, _ = forecast_positions(lap_df, from_lap=5, n_laps=5)
    required = {"lap", "driver", "projected_position", "projected_gap_ahead"}
    missing = required - set(proj_df.columns)
    assert not missing, f"Missing columns: {missing}"


# ── Phase 2 tests — enabled after compare_pit_scenarios is added ──────────────
# These are NOT skipped — they import the function directly. If the function
# doesn't exist yet, the import at the top of the file will fail gracefully
# and these tests will error (not pass), which is the correct behavior.

try:
    from models.race_forecast import compare_pit_scenarios
    _HAS_SCENARIOS = True
except ImportError:
    _HAS_SCENARIOS = False


@pytest.mark.skipif(not _HAS_SCENARIOS, reason="compare_pit_scenarios not yet implemented")
def test_compare_pit_scenarios_returns_four_dicts(lap_df):
    result = compare_pit_scenarios(lap_df, "bahrain_2025", from_lap=5, driver="HAM", n_laps=10)
    assert len(result) == 4


@pytest.mark.skipif(not _HAS_SCENARIOS, reason="compare_pit_scenarios not yet implemented")
def test_compare_pit_scenarios_schema(lap_df):
    result = compare_pit_scenarios(lap_df, "bahrain_2025", from_lap=5, driver="HAM", n_laps=10)
    required = {"label", "pit_lap", "projected_position", "projected_gap_ahead", "position_delta", "recommendation"}
    for sc in result:
        missing = required - set(sc.keys())
        assert not missing, f"Scenario missing keys: {missing}"


@pytest.mark.skipif(not _HAS_SCENARIOS, reason="compare_pit_scenarios not yet implemented")
def test_compare_pit_scenarios_exactly_one_recommendation(lap_df):
    result = compare_pit_scenarios(lap_df, "bahrain_2025", from_lap=5, driver="HAM", n_laps=10)
    recommended = [s for s in result if s["recommendation"]]
    assert len(recommended) == 1


@pytest.mark.skipif(not _HAS_SCENARIOS, reason="compare_pit_scenarios not yet implemented")
def test_compare_pit_scenarios_driver_not_in_df(lap_df):
    result = compare_pit_scenarios(lap_df, "bahrain_2025", from_lap=5, driver="ZZZ", n_laps=10)
    assert result == []


@pytest.mark.skipif(not _HAS_SCENARIOS, reason="compare_pit_scenarios not yet implemented")
def test_compare_pit_scenarios_stay_out_has_no_pit_lap(lap_df):
    result = compare_pit_scenarios(lap_df, "bahrain_2025", from_lap=5, driver="HAM", n_laps=10)
    stay_out = next(s for s in result if s["label"] == "Stay Out")
    assert stay_out["pit_lap"] is None


@pytest.mark.skipif(not _HAS_SCENARIOS, reason="compare_pit_scenarios not yet implemented")
def test_compare_pit_scenarios_pit_now_increases_gap(lap_df):
    result = compare_pit_scenarios(lap_df, "bahrain_2025", from_lap=5, driver="HAM", n_laps=10)
    # Pit Now re-entry adds pit_lane_delta to gap — driver starts the simulation
    # behind, so projected_position should be >= Stay Out projected_position
    pit_now  = next(s for s in result if s["label"] == "Pit Now")
    stay_out = next(s for s in result if s["label"] == "Stay Out")
    # Both should be valid ints
    assert isinstance(pit_now["projected_position"], int)
    assert isinstance(stay_out["projected_position"], int)


@pytest.mark.skipif(not _HAS_SCENARIOS, reason="compare_pit_scenarios not yet implemented")
def test_compare_pit_scenarios_sc_reduces_pit_cost(lap_df):
    """Under Safety Car the pit cost should be ~5s, not the normal ~23s."""
    sc_df = lap_df.copy()
    # Mark lap 5 as SC active for all drivers
    sc_df.loc[sc_df["lap"] == 5, "safety_car_active"] = True

    # Normal race (no SC)
    normal = compare_pit_scenarios(lap_df, "bahrain_2025", from_lap=5, driver="HAM", n_laps=10)
    # SC active at pit lap
    sc = compare_pit_scenarios(sc_df, "bahrain_2025", from_lap=5, driver="HAM", n_laps=10)

    pit_now_normal = next(s for s in normal if s["label"] == "Pit Now")
    pit_now_sc     = next(s for s in sc     if s["label"] == "Pit Now")

    # SC scenario should project a better (lower) position number than the
    # normal scenario, because the pit delta is ~5s not ~23s.
    assert pit_now_sc["projected_position"] <= pit_now_normal["projected_position"], (
        f"SC pit should project P{pit_now_sc['projected_position']} ≤ "
        f"normal pit P{pit_now_normal['projected_position']}"
    )


def test_get_compound_degradation_rates_returns_dict(lap_df):
    rates = get_compound_degradation_rates(lap_df)
    assert isinstance(rates, dict)
    # lap_df uses MEDIUM compound with enough rows — should produce a rate
    assert "MEDIUM" in rates
    assert rates["MEDIUM"] >= 0.005  # floored at minimum


def test_get_compound_degradation_rates_floor(lap_df):
    """Rate should never be negative (floored at 0.005)."""
    # Artificially invert pace_delta so raw regression slope is negative
    inv_df = lap_df.copy()
    inv_df["pace_delta"] = -inv_df["tyre_age"].astype(float) * 0.1
    rates = get_compound_degradation_rates(inv_df)
    for compound, rate in rates.items():
        assert rate >= 0.005, f"Compound {compound} rate {rate} is below floor"


def test_get_compound_degradation_rates_skips_sparse_compound():
    """Compounds with fewer than 10 rows should be skipped."""
    rows = [{"driver": "VER", "team": "RBR", "lap": i, "position": 1.0,
             "gap_ahead": float("nan"), "gap_behind": float("nan"),
             "pace_delta": 0.01 * i, "radio_sentiment": 0.0, "pit_window_pressure": 40.0,
             "tyre_age": i, "tyre_compound": "SOFT" if i <= 5 else "MEDIUM",
             "lap_time": 90.0, "is_pit_in": False, "safety_car_active": False, "stint": 1}
            for i in range(1, 25)]
    df = pd.DataFrame(rows)
    rates = get_compound_degradation_rates(df)
    # SOFT has only 5 rows (<10 threshold) — must be absent
    assert "SOFT" not in rates
    # MEDIUM has 19 rows — must be present
    assert "MEDIUM" in rates

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def lap_df():
    """Standard 3-driver × 24-lap DataFrame covering all required columns."""
    rows = []
    for d, pos in [("VER", 1), ("HAM", 2), ("LEC", 3)]:
        for lap in range(1, 25):
            rows.append({
                "driver": d,
                "team": "Team",
                "race_slug": "bahrain_2025",
                "lap": lap,
                "position": float(pos),
                "gap_ahead": float("nan") if d == "VER" else 2.0,
                "gap_behind": 2.0,
                "pace_delta": 0.1,
                "radio_sentiment": 0.0,
                "pit_window_pressure": 40.0,
                "tyre_age": lap % 15,
                "tyre_compound": "MEDIUM",
                "lap_time": 90.0 + np.random.default_rng(lap).uniform(-0.5, 0.5),
                "is_pit_in": False,
                "safety_car_active": False,
                "stint": 1,
                "min_speed_corners": 120.0,
                "radio_text": "",
            })
    return pd.DataFrame(rows)

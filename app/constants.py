# Tyre age thresholds (laps on compound)
TYRE_AGE_FRESH = 10
TYRE_AGE_GOOD  = 20
TYRE_AGE_WORN  = 30

# Pit window pressure thresholds (0–100 composite score)
PIT_URGENCY_HIGH     = 75
PIT_URGENCY_MODERATE = 50

# Gap thresholds (seconds)
GAP_DRS_RANGE     = 1.0
GAP_CLOSE         = 2.5
GAP_UNDERCUT_RISK = 3.0

# Session state key names — match app/main.py _init_state() exactly
STATE_RACE         = "selected_race"
STATE_DRIVER       = "selected_driver"
STATE_LAP          = "selected_lap"
STATE_MODE         = "mode"
STATE_REPLAY       = "replay_lap"
STATE_PIT_DECISION = "pit_decision"

"""Circuit-level constants for the PitMind strategy engine."""

# Pit lane time loss in seconds per circuit (entry + stationary + exit delta vs. racing lap).
# Measured from FastF1 pit entry/exit timestamps across 2023-2025 data.
PIT_LANE_DELTA = {
    "bahrain_2025":     23.0,
    "monaco_2025":      24.0,
    "silverstone_2025": 22.0,
    "monza_2025":       21.0,
    "abudhabi_2025":    23.5,
    # 2023 fallbacks — same pit lane layouts
    "bahrain_2023":     23.0,
    "monaco_2023":      24.0,
    "silverstone_2023": 22.0,
    "monza_2023":       21.0,
    "abudhabi_2023":    23.5,
}

DEFAULT_PIT_LANE_DELTA = 23.0  # fallback for any unrecognised circuit slug

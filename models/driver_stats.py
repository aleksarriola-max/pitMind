"""Season-level driver statistics aggregated across all pre-computed races."""

import numpy as np
import pandas as pd

TRAIT_COLS = [
    "overtake_tendency", "position_vulnerability", "aggression_level",
    "tyre_preservation", "pit_compliance", "restart_aggression",
    "pressure_consistency", "late_braking_tendency", "radio_stress_frequency",
    "undercut_susceptibility", "dirty_air_tolerance",
]


def compute_season_stats(all_race_dfs: dict) -> pd.DataFrame:
    """
    Aggregate per-driver stats across all races.

    all_race_dfs: {slug: DataFrame}
    Returns DataFrame indexed by driver with one row per driver.
    """
    rows = []
    for slug, df in all_race_dfs.items():
        if len(df) == 0:
            continue
        race_name = df["race"].iloc[0] if "race" in df.columns else slug
        for driver, grp in df.groupby("driver"):
            grp = grp.sort_values("lap")
            final_pos = grp["position"].dropna()
            grid_pos = grp["grid_position"].dropna()
            lap_times = grp["lap_time"].dropna()

            row = {
                "driver": driver,
                "race": race_name,
                "slug": slug,
                "finish_position": int(final_pos.iloc[-1]) if len(final_pos) > 0 else None,
                "grid_position": int(grid_pos.iloc[0]) if len(grid_pos) > 0 else None,
                "positions_gained": (int(grid_pos.iloc[0]) - int(final_pos.iloc[-1]))
                                    if len(grid_pos) > 0 and len(final_pos) > 0 else 0,
                "best_lap_time": float(lap_times.min()) if len(lap_times) > 0 else None,
                "avg_pace_delta": float(grp["pace_delta"].dropna().mean()) if "pace_delta" in grp.columns else None,
            }
            # Trait scores (per-race averages for this driver)
            for col in TRAIT_COLS:
                if col in grp.columns and grp[col].notna().any():
                    row[col] = float(grp[col].dropna().mean())
                else:
                    row[col] = None
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    long_df = pd.DataFrame(rows)

    # Aggregate across races per driver
    agg = {}
    for driver, grp in long_df.groupby("driver"):
        d = {"driver": driver}
        d["races_counted"] = len(grp)
        d["avg_finish"] = round(grp["finish_position"].dropna().mean(), 1) if grp["finish_position"].notna().any() else None
        d["avg_grid"] = round(grp["grid_position"].dropna().mean(), 1) if grp["grid_position"].notna().any() else None
        d["avg_positions_gained"] = round(grp["positions_gained"].mean(), 1)
        d["best_lap_time"] = round(grp["best_lap_time"].dropna().min(), 3) if grp["best_lap_time"].notna().any() else None
        d["avg_pace_delta"] = round(grp["avg_pace_delta"].dropna().mean(), 3) if grp["avg_pace_delta"].notna().any() else None

        for col in TRAIT_COLS:
            vals = grp[col].dropna()
            d[col] = round(float(vals.mean()), 1) if len(vals) > 0 else None

        agg[driver] = d

    stats_df = pd.DataFrame(list(agg.values())).set_index("driver")
    return stats_df


def season_arc(all_race_dfs: dict, driver: str) -> pd.DataFrame:
    """
    Per-race metrics for a single driver — used for the season arc line charts.
    Returns DataFrame: race, finish_position, avg_pace_delta, + trait cols.
    """
    rows = []
    race_order = list(all_race_dfs.keys())
    for slug in race_order:
        df = all_race_dfs.get(slug, pd.DataFrame())
        if len(df) == 0:
            continue
        grp = df[df["driver"] == driver].sort_values("lap")
        if len(grp) == 0:
            continue
        race_name = grp["race"].iloc[0] if "race" in grp.columns else slug
        final_pos = grp["position"].dropna()
        row = {
            "race": race_name,
            "slug": slug,
            "finish_position": int(final_pos.iloc[-1]) if len(final_pos) > 0 else None,
            "avg_pace_delta": round(float(grp["pace_delta"].dropna().mean()), 3) if "pace_delta" in grp.columns else None,
        }
        for col in TRAIT_COLS:
            if col in grp.columns and grp[col].notna().any():
                row[col] = round(float(grp[col].dropna().mean()), 1)
            else:
                row[col] = None
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def rank_drivers(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Add rank columns (1=best) for key metrics."""
    if len(stats_df) == 0:
        return stats_df

    ranked = stats_df.copy()
    # Lower finish/grid = better
    for col in ["avg_finish", "avg_grid", "avg_pace_delta", "best_lap_time"]:
        if col in ranked.columns:
            ranked[f"{col}_rank"] = ranked[col].rank(ascending=True, na_option="bottom")
    # Higher trait = better (most traits)
    for col in ["overtake_tendency", "aggression_level", "tyre_preservation",
                "pit_compliance", "restart_aggression", "pressure_consistency",
                "dirty_air_tolerance", "avg_positions_gained"]:
        if col in ranked.columns:
            ranked[f"{col}_rank"] = ranked[col].rank(ascending=False, na_option="bottom")
    return ranked

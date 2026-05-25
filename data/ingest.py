"""FastF1 + OpenF1 unified ingestion pipeline.

Produces a per-lap DataFrame matching the PitMind schema for a given race.
Call build_race_dataframe() to get the unified DataFrame.
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import fastf1

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
FASTF1_CACHE_DIR = os.path.join(CACHE_DIR, "fastf1")
OPENF1_BASE = "https://api.openf1.org/v1"

RACES = {
    "bahrain_2023":     {"year": 2023, "round": 1,  "name": "Bahrain 2023"},
    "monaco_2023":      {"year": 2023, "round": 6,  "name": "Monaco 2023"},
    "silverstone_2023": {"year": 2023, "round": 10, "name": "Silverstone 2023"},
    "monza_2023":       {"year": 2023, "round": 14, "name": "Monza 2023"},
    "abudhabi_2023":    {"year": 2023, "round": 22, "name": "Abu Dhabi 2023"},
}

DRIVERS = ["VER", "HAM", "LEC", "SAI", "ALO", "PER", "NOR", "RUS"]

os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)


# ---------------------------------------------------------------------------
# FastF1 pull
# ---------------------------------------------------------------------------

def pull_fastf1(year: int, round_num: int) -> dict:
    """Pull lap times, telemetry, tyre data, sector splits from FastF1."""
    log.info(f"Loading FastF1 session {year} round {round_num}")
    session = fastf1.get_session(year, round_num, "R")
    session.load(telemetry=True, weather=True, messages=True)

    laps = session.laps.copy()
    laps = laps[laps["Driver"].isin(DRIVERS)].reset_index(drop=True)

    lap_rows = []
    for _, lap in laps.iterrows():
        driver = lap["Driver"]
        team = lap["Team"] if pd.notna(lap.get("Team")) else ""
        lap_num = int(lap["LapNumber"]) if pd.notna(lap["LapNumber"]) else None
        if lap_num is None:
            continue

        lap_time = lap["LapTime"].total_seconds() if pd.notna(lap["LapTime"]) else np.nan
        tyre_compound = lap.get("Compound", "UNKNOWN")
        tyre_age = int(lap["TyreLife"]) if pd.notna(lap.get("TyreLife")) else 0
        stint = int(lap["Stint"]) if pd.notna(lap.get("Stint")) else 0
        position = int(lap["Position"]) if pd.notna(lap.get("Position")) else np.nan
        sector1 = lap["Sector1Time"].total_seconds() if pd.notna(lap.get("Sector1Time")) else np.nan
        sector2 = lap["Sector2Time"].total_seconds() if pd.notna(lap.get("Sector2Time")) else np.nan
        sector3 = lap["Sector3Time"].total_seconds() if pd.notna(lap.get("Sector3Time")) else np.nan
        is_pit_out = bool(lap.get("PitOutTime") is not None and pd.notna(lap.get("PitOutTime")))
        is_pit_in = bool(lap.get("PitInTime") is not None and pd.notna(lap.get("PitInTime")))

        # Braking telemetry summary for driver soul features
        try:
            tel = lap.get_telemetry()
            min_speed_corners = float(tel["Speed"].quantile(0.05)) if len(tel) > 0 else np.nan
            max_throttle = float(tel["Throttle"].max()) if len(tel) > 0 else np.nan
            drs_used = bool((tel["DRS"] > 8).any()) if "DRS" in tel.columns and len(tel) > 0 else False
        except Exception:
            min_speed_corners = np.nan
            max_throttle = np.nan
            drs_used = False

        lap_rows.append({
            "driver": driver,
            "team": team,
            "lap": lap_num,
            "lap_time": lap_time,
            "tyre_compound": tyre_compound,
            "tyre_age": tyre_age,
            "stint": stint,
            "position": position,
            "sector1_time": sector1,
            "sector2_time": sector2,
            "sector3_time": sector3,
            "is_pit_out": is_pit_out,
            "is_pit_in": is_pit_in,
            "min_speed_corners": min_speed_corners,
            "max_throttle": max_throttle,
            "drs_used": drs_used,
        })

    lap_df = pd.DataFrame(lap_rows)

    # Compute pace_delta: lap_time - driver's personal best this stint
    def compute_pace_delta(group):
        best = group["lap_time"].min()
        group = group.copy()
        group["pace_delta"] = group["lap_time"] - best
        return group

    lap_df = lap_df.groupby(["driver", "stint"], group_keys=False).apply(compute_pace_delta)

    # Weather per lap (approximate from session weather data)
    if hasattr(session, "weather_data") and session.weather_data is not None and len(session.weather_data) > 0:
        weather = session.weather_data[["Time", "TrackTemp", "Rainfall"]].copy()
        weather["Time"] = weather["Time"].dt.total_seconds()
        lap_df["weather_track_temp"] = lap_df["lap"].apply(
            lambda l: _approx_weather(weather, l, session)
        )
    else:
        lap_df["weather_track_temp"] = np.nan

    # Safety car laps from session messages
    sc_laps = _extract_sc_laps(session)
    lap_df["safety_car_active"] = lap_df["lap"].isin(sc_laps)

    return {"lap_df": lap_df, "session": session}


def _approx_weather(weather_df: pd.DataFrame, lap_num: int, session) -> float:
    """Map a lap number to a track temp reading (rough approximation)."""
    try:
        total_laps = session.laps["LapNumber"].max()
        total_session_time = weather_df["Time"].max()
        approx_time = (lap_num / total_laps) * total_session_time
        idx = (weather_df["Time"] - approx_time).abs().idxmin()
        return float(weather_df.loc[idx, "TrackTemp"])
    except Exception:
        return np.nan


def _extract_sc_laps(session) -> set:
    """Extract lap numbers where a safety car was deployed."""
    sc_laps = set()
    try:
        messages = session.race_control_messages
        if messages is None or len(messages) == 0:
            return sc_laps
        sc_msgs = messages[messages["Message"].str.contains("SAFETY CAR", case=False, na=False)]
        end_msgs = messages[messages["Message"].str.contains("SAFETY CAR IN THIS LAP|GREEN LIGHT", case=False, na=False)]
        if len(sc_msgs) == 0:
            return sc_laps
        # Rough: mark all laps between first SC message and first end message
        first_sc_lap = int(sc_msgs["Lap"].iloc[0]) if "Lap" in sc_msgs.columns else 0
        if len(end_msgs) > 0:
            end_lap = int(end_msgs["Lap"].iloc[0]) if "Lap" in end_msgs.columns else first_sc_lap + 5
        else:
            end_lap = first_sc_lap + 5
        sc_laps = set(range(first_sc_lap, end_lap + 1))
    except Exception:
        pass
    return sc_laps


# ---------------------------------------------------------------------------
# OpenF1 pull
# ---------------------------------------------------------------------------

OPENF1_MEETING_KEYS = {
    "bahrain_2023":     1141,
    "monaco_2023":      1210,
    "silverstone_2023": 1214,
    "monza_2023":       1218,
    "abudhabi_2023":    1226,
}

OPENF1_SESSION_KEYS = {
    "bahrain_2023":     7953,
    "monaco_2023":      9094,
    "silverstone_2023": 9126,
    "monza_2023":       9157,
    "abudhabi_2023":    9197,
}

OPENF1_DRIVER_NUMS = {
    "VER": "1", "HAM": "44", "LEC": "16", "SAI": "55",
    "ALO": "14", "PER": "11", "NOR": "4", "RUS": "63",
}


def _openf1_get(endpoint: str, params: dict, retries: int = 3) -> list:
    """GET from OpenF1 with retry logic."""
    url = f"{OPENF1_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                log.warning(f"OpenF1 {endpoint} attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                log.warning(f"OpenF1 {endpoint} failed after {retries} attempts: {e}")
                return []


def pull_openf1(slug: str) -> dict:
    """Pull interval, position, race_control, and weather data from OpenF1."""
    session_key = OPENF1_SESSION_KEYS.get(slug)
    if session_key is None:
        log.warning(f"No OpenF1 session key for {slug}")
        return {"intervals": pd.DataFrame(), "position": pd.DataFrame(),
                "race_control": pd.DataFrame(), "weather": pd.DataFrame()}

    log.info(f"Pulling OpenF1 data for {slug} (session_key={session_key})")

    intervals_raw = _openf1_get("intervals", {"session_key": session_key})
    position_raw = _openf1_get("position", {"session_key": session_key})
    race_ctrl_raw = _openf1_get("race_control", {"session_key": session_key})
    weather_raw = _openf1_get("weather", {"session_key": session_key})

    intervals_df = pd.DataFrame(intervals_raw) if intervals_raw else pd.DataFrame()
    position_df = pd.DataFrame(position_raw) if position_raw else pd.DataFrame()
    race_ctrl_df = pd.DataFrame(race_ctrl_raw) if race_ctrl_raw else pd.DataFrame()
    weather_df = pd.DataFrame(weather_raw) if weather_raw else pd.DataFrame()

    return {
        "intervals": intervals_df,
        "position": position_df,
        "race_control": race_ctrl_df,
        "weather": weather_df,
    }


# ---------------------------------------------------------------------------
# Merge FastF1 + OpenF1 → unified per-lap DataFrame
# ---------------------------------------------------------------------------

def merge_to_lap_df(fastf1_result: dict, openf1_result: dict, race_name: str) -> pd.DataFrame:
    """Merge FastF1 and OpenF1 data into the unified per-lap schema."""
    df = fastf1_result["lap_df"].copy()
    df["race"] = race_name

    # --- gap_ahead / gap_behind from OpenF1 intervals (timestamp-based merge) ---
    intervals = openf1_result["intervals"]
    if len(intervals) > 0 and "driver_number" in intervals.columns and "date" in intervals.columns:
        num_to_abbr = {v: k for k, v in OPENF1_DRIVER_NUMS.items()}
        intervals = intervals.copy()
        intervals["driver"] = intervals["driver_number"].astype(str).map(num_to_abbr)
        intervals = intervals.dropna(subset=["driver"])
        intervals["date"] = pd.to_datetime(intervals["date"], format="ISO8601", utc=True)
        intervals["interval_sec"] = pd.to_numeric(intervals["interval"], errors="coerce")
        intervals["gap_to_leader_sec"] = pd.to_numeric(intervals["gap_to_leader"], errors="coerce")

        # Get session start time from FastF1 lap Time column (session-relative seconds)
        # We map lap number to session time by using the cumulative lap Time from FastF1
        session = fastf1_result["session"]
        laps_raw = session.laps[session.laps["Driver"].isin(DRIVERS)].copy()
        laps_raw = laps_raw[["Driver", "LapNumber", "Time"]].dropna()
        laps_raw["session_sec"] = laps_raw["Time"].dt.total_seconds()

        # Approximate session wall-clock start from OpenF1 interval dates
        session_start = intervals["date"].min()

        # For each driver+lap, interpolate gap from the nearest interval record
        gap_lookup = {}
        for driver, grp in intervals.groupby("driver"):
            grp = grp.sort_values("date")
            grp["elapsed_sec"] = (grp["date"] - session_start).dt.total_seconds()
            driver_laps = laps_raw[laps_raw["Driver"] == driver].sort_values("session_sec")
            for _, row in driver_laps.iterrows():
                lap_num = int(row["LapNumber"])
                lap_sec = row["session_sec"]
                # Nearest interval record by elapsed session time
                diff = (grp["elapsed_sec"] - lap_sec).abs()
                if len(diff) > 0:
                    nearest = grp.loc[diff.idxmin()]
                    gap_lookup[(driver, lap_num)] = {
                        "gap_ahead": nearest["interval_sec"],
                        "interval": nearest["interval_sec"],
                    }

        if gap_lookup:
            df["gap_ahead"] = df.apply(
                lambda r: gap_lookup.get((r["driver"], r["lap"]), {}).get("gap_ahead", np.nan), axis=1
            )
            df["interval"] = df.apply(
                lambda r: gap_lookup.get((r["driver"], r["lap"]), {}).get("interval", np.nan), axis=1
            )
        else:
            df["gap_ahead"] = np.nan
            df["interval"] = np.nan

        # Derive gap_behind by looking at the car directly behind in each lap
        df = df.sort_values(["lap", "position"])
        df["gap_behind"] = df.groupby("lap")["gap_ahead"].shift(-1)
        df["gap_ahead"] = pd.to_numeric(df["gap_ahead"], errors="coerce")
        df["gap_behind"] = pd.to_numeric(df["gap_behind"], errors="coerce")
    else:
        df["gap_ahead"] = np.nan
        df["gap_behind"] = np.nan
        df["interval"] = np.nan

    # --- race_control: safety car, flags ---
    race_ctrl = openf1_result["race_control"]
    if len(race_ctrl) > 0 and "lap_number" in race_ctrl.columns:
        sc_laps_openf1 = set(
            race_ctrl[race_ctrl.get("category", pd.Series()).eq("SafetyCar") |
                       race_ctrl.get("message", pd.Series()).str.contains("SAFETY CAR", na=False)
                       ]["lap_number"].dropna().astype(int).tolist()
        ) if "message" in race_ctrl.columns or "category" in race_ctrl.columns else set()
        df["safety_car_active"] = df["safety_car_active"] | df["lap"].isin(sc_laps_openf1)

        # SC/VSC/flag events for Momentum Map overlays — store as JSON-serializable list
        flag_events = race_ctrl[["lap_number", "message", "flag"]].dropna(subset=["lap_number"]).rename(
            columns={"lap_number": "lap"}
        ) if "message" in race_ctrl.columns else pd.DataFrame()
    else:
        flag_events = pd.DataFrame()

    # --- weather supplement ---
    weather = openf1_result["weather"]
    if len(weather) > 0 and "track_temperature" in weather.columns and df["weather_track_temp"].isna().all():
        avg_track_temp = float(weather["track_temperature"].mean())
        df["weather_track_temp"] = avg_track_temp

    # --- pit_window_pressure: composite 0-100 ---
    df["pit_window_pressure"] = _compute_pit_pressure(df)

    # Initialize columns that will be filled by later pipeline steps
    for col in ["radio_text", "radio_sentiment", "momentum_score",
                "overtake_tendency", "position_vulnerability", "aggression_level",
                "tyre_preservation", "pit_compliance", "restart_aggression",
                "pressure_consistency", "late_braking_tendency", "radio_stress_frequency",
                "undercut_susceptibility", "dirty_air_tolerance",
                "position_gain_prob", "position_loss_prob", "pit_prob", "incident_risk"]:
        if col not in df.columns:
            df[col] = np.nan if col.endswith(("_prob", "_risk", "_score", "_sentiment")) else (
                "" if col in ("radio_text",) else np.nan
            )

    df = df.sort_values(["driver", "lap"]).reset_index(drop=True)
    return df


def _compute_pit_pressure(df: pd.DataFrame) -> pd.Series:
    """
    Pit window pressure (0-100): rises with tyre age and pace degradation,
    spikes when gap_ahead is small (undercut threat) or gap_behind is closing.
    """
    age_score = (df["tyre_age"].clip(0, 40) / 40 * 60).fillna(30)
    pace_score = (df["pace_delta"].clip(0, 3) / 3 * 30).fillna(0)
    gap_score = ((1 - df["gap_ahead"].clip(0, 5) / 5) * 10).fillna(5)
    pressure = (age_score + pace_score + gap_score).clip(0, 100)
    return pressure


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_race_dataframe(slug: str) -> pd.DataFrame:
    """Full pipeline: FastF1 + OpenF1 → merged per-lap DataFrame."""
    race_info = RACES[slug]
    fastf1_result = pull_fastf1(race_info["year"], race_info["round"])
    openf1_result = pull_openf1(slug)
    df = merge_to_lap_df(fastf1_result, openf1_result, race_info["name"])
    return df


def save_race(df: pd.DataFrame, slug: str) -> str:
    """Serialize the per-lap DataFrame to parquet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{slug}.parquet")
    df.to_parquet(path, index=False)
    log.info(f"Saved {len(df)} rows to {path}")
    return path


def load_race(slug: str) -> pd.DataFrame:
    """Load a pre-computed parquet file."""
    path = os.path.join(CACHE_DIR, f"{slug}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached data for {slug}. Run build_data.py first.")
    return pd.read_parquet(path)

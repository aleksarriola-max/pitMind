"""FastF1 + OpenF1 unified ingestion pipeline.

Produces a per-lap DataFrame matching the PitMind schema for a given race.
Call build_race_dataframe() to get the unified DataFrame.
"""

import os
import json
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
    "bahrain_2025":     {"year": 2025, "round": 4,  "name": "Bahrain 2025",   "openf1_circuit": "bahrain"},
    "monaco_2025":      {"year": 2025, "round": 8,  "name": "Monaco 2025",    "openf1_circuit": "monaco"},
    "silverstone_2025": {"year": 2025, "round": 12, "name": "British 2025",   "openf1_circuit": "silverstone"},
    "monza_2025":       {"year": 2025, "round": 16, "name": "Italian 2025",   "openf1_circuit": "monza"},
    "abudhabi_2025":    {"year": 2025, "round": 24, "name": "Abu Dhabi 2025", "openf1_circuit": "abu_dhabi"},
}

DRIVERS = ["VER", "HAM", "LEC", "PIA", "ALO", "OCO", "NOR", "RUS"]

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

    return {"lap_df": lap_df, "session": session, "year": year, "round_num": round_num}


def pull_qualifying(year: int, round_num: int) -> dict:
    """Pull qualifying results and return {driver_code: grid_position} mapping."""
    try:
        session = fastf1.get_session(year, round_num, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps.copy()
        laps = laps[laps["Driver"].isin(DRIVERS) & laps["LapTime"].notna()]
        best = laps.groupby("Driver")["LapTime"].min().reset_index()
        best = best.dropna(subset=["LapTime"]).sort_values("LapTime").reset_index(drop=True)
        result = {row["Driver"]: int(i + 1) for i, (_, row) in enumerate(best.iterrows())}
        log.info(f"Qualifying grid: {result}")
        return result
    except Exception as e:
        log.warning(f"Could not load qualifying for {year} round {round_num}: {e}")
        return {}


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

OPENF1_DRIVER_NUMS = {
    "VER": "1", "HAM": "44", "LEC": "16", "SAI": "55",
    "ALO": "14", "PER": "11", "NOR": "4", "RUS": "63",
}

_session_key_cache: dict[str, int] = {}


def _get_openf1_session_key(slug: str) -> int | None:
    """Auto-lookup OpenF1 session key by querying the /sessions endpoint."""
    if slug in _session_key_cache:
        return _session_key_cache[slug]

    race_info = RACES.get(slug, {})
    year = race_info.get("year")
    circuit = race_info.get("openf1_circuit", "")
    if not year or not circuit:
        return None

    sessions = _openf1_get("sessions", {"year": year, "session_name": "Race"})
    for s in sessions:
        key = s.get("circuit_short_name", "").lower().replace("-", "_").replace(" ", "_")
        if circuit.lower() in key or key in circuit.lower():
            sk = s.get("session_key")
            if sk:
                _session_key_cache[slug] = int(sk)
                log.info(f"Auto-resolved OpenF1 session_key for {slug}: {sk}")
                return int(sk)

    log.warning(f"Could not auto-resolve OpenF1 session_key for {slug} (circuit='{circuit}', year={year})")
    return None


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
    session_key = _get_openf1_session_key(slug)
    if session_key is None:
        log.warning(f"No OpenF1 session key for {slug}, skipping OpenF1 data")
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

def merge_to_lap_df(fastf1_result: dict, openf1_result: dict, race_name: str,
                    slug: str = "", grid_positions: dict | None = None) -> pd.DataFrame:
    """Merge FastF1 and OpenF1 data into the unified per-lap schema."""
    df = fastf1_result["lap_df"].copy()
    df["race"] = race_name
    df["race_slug"] = slug

    # Grid position from qualifying (constant per driver per race)
    if grid_positions:
        df["grid_position"] = df["driver"].map(grid_positions).fillna(10).astype(int)
    else:
        df["grid_position"] = 10  # fallback midfield

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
    grid_positions = pull_qualifying(race_info["year"], race_info["round"])
    df = merge_to_lap_df(fastf1_result, openf1_result, race_info["name"], slug=slug, grid_positions=grid_positions)
    return df


def save_race(df: pd.DataFrame, slug: str) -> str:
    """Serialize the per-lap DataFrame to parquet (atomic write)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{slug}.parquet")
    tmp_path = path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    log.info(f"Saved {len(df)} rows to {path}")
    return path


def load_race(slug: str) -> pd.DataFrame:
    """Load a pre-computed parquet file."""
    path = os.path.join(CACHE_DIR, f"{slug}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached data for {slug}. Run build_data.py first.")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Flag / Race Control helpers
# ---------------------------------------------------------------------------

def _build_flag_periods(race_ctrl_df: pd.DataFrame) -> list:
    """Convert race_control rows into consolidated flag period dicts."""
    if len(race_ctrl_df) == 0 or "lap_number" not in race_ctrl_df.columns:
        return []

    rc = race_ctrl_df.copy()
    rc["lap_number"] = pd.to_numeric(rc["lap_number"], errors="coerce")
    rc = rc.dropna(subset=["lap_number"])
    rc["lap_number"] = rc["lap_number"].astype(int)
    periods = []

    msg_col = "message" if "message" in rc.columns else None

    if msg_col:
        # Safety Car
        sc_starts = rc[rc[msg_col].str.contains("SAFETY CAR DEPLOYED|SAFETY CAR OUT", case=False, na=False)]
        sc_ends   = rc[rc[msg_col].str.contains("SAFETY CAR IN THIS LAP|SAFETY CAR WITHDRAWN|GREEN LIGHT|TRACK CLEAR", case=False, na=False)]
        for _, row in sc_starts.iterrows():
            lap_s = int(row["lap_number"])
            after = sc_ends[sc_ends["lap_number"] > lap_s]
            lap_e = int(after["lap_number"].min()) if len(after) > 0 else lap_s + 5
            periods.append({"flag": "SAFETY_CAR", "lap_start": lap_s, "lap_end": lap_e})

        # Virtual Safety Car
        vsc_starts = rc[rc[msg_col].str.contains("VIRTUAL SAFETY CAR DEPLOYED", case=False, na=False)]
        vsc_ends   = rc[rc[msg_col].str.contains("VIRTUAL SAFETY CAR ENDING|VIRTUAL SAFETY CAR WITHDRAWN", case=False, na=False)]
        for _, row in vsc_starts.iterrows():
            lap_s = int(row["lap_number"])
            after = vsc_ends[vsc_ends["lap_number"] > lap_s]
            lap_e = int(after["lap_number"].min()) if len(after) > 0 else lap_s + 3
            periods.append({"flag": "VIRTUAL_SAFETY_CAR", "lap_start": lap_s, "lap_end": lap_e})

    # Yellow / Red from flag column
    if "flag" in rc.columns:
        for flag_val, label in [("YELLOW", "YELLOW"), ("DOUBLE YELLOW", "DOUBLE_YELLOW"), ("RED", "RED")]:
            rows = rc[rc["flag"] == flag_val]
            for _, row in rows.iterrows():
                lap_n = int(row["lap_number"])
                periods.append({"flag": label, "lap_start": lap_n, "lap_end": lap_n + 1})

    return periods


def save_flags(flag_periods: list, slug: str) -> str:
    """Save flag periods list to JSON."""
    path = os.path.join(CACHE_DIR, f"{slug}_flags.json")
    with open(path, "w") as f:
        json.dump(flag_periods, f, indent=2)
    log.info(f"Saved {len(flag_periods)} flag periods to {path}")
    return path


def load_flags(slug: str, df: "pd.DataFrame | None" = None) -> list:
    """
    Load flag periods for a race.
    Prefers pre-saved JSON; falls back to deriving SC from safety_car_active column.
    """
    json_path = os.path.join(CACHE_DIR, f"{slug}_flags.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)

    # Fallback: derive SC periods from safety_car_active column
    if df is not None and "safety_car_active" in df.columns:
        sc_laps = sorted(df[df["safety_car_active"]]["lap"].unique())
        if not sc_laps:
            return []
        periods, start, prev = [], sc_laps[0], sc_laps[0]
        for lap in sc_laps[1:]:
            if lap > prev + 1:
                periods.append({"flag": "SAFETY_CAR", "lap_start": int(start), "lap_end": int(prev)})
                start = lap
            prev = lap
        periods.append({"flag": "SAFETY_CAR", "lap_start": int(start), "lap_end": int(prev)})
        return periods

    return []


def build_flags(slug: str) -> list:
    """Pull race_control from OpenF1, derive flag periods, and save to JSON."""
    openf1_result = pull_openf1(slug)
    periods = _build_flag_periods(openf1_result.get("race_control", pd.DataFrame()))
    save_flags(periods, slug)
    return periods

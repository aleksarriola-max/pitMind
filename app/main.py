"""PitMind — Streamlit entry point.

Manages sidebar, session state, and tab routing between the three views.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from data.ingest import RACES, DRIVERS, load_race
from models.momentum import load_shifts
from models.driver_soul import TRAIT_COLS, PRED_COLS

st.set_page_config(
    page_title="PitMind — F1 AI Agent",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_APP_VERSION = "3.4"  # bump to force session state reset on redeploy

def _init_state():
    # Clear stale session state from old deployments
    if st.session_state.get("_version") != _APP_VERSION:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state["_version"] = _APP_VERSION

    defaults = {
        "selected_race": "bahrain_2025",
        "selected_driver": "VER",
        "selected_lap": 1,
        "mode": "fan",
        "replay_lap": None,
        "pit_decision": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# Restore state from URL query params (enables shareable links)
_qp = st.query_params
if "race" in _qp and _qp["race"] in RACES:
    st.session_state.setdefault("selected_race", _qp["race"])
    if st.session_state.get("selected_race") != _qp["race"]:
        st.session_state.selected_race = _qp["race"]
if "driver" in _qp and _qp["driver"] in DRIVERS:
    st.session_state.setdefault("selected_driver", _qp["driver"])
if "lap" in _qp:
    try:
        st.session_state.setdefault("selected_lap", int(_qp["lap"]))
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading race data...")
def get_race_df(slug: str) -> pd.DataFrame:
    try:
        return load_race(slug)
    except FileNotFoundError:
        st.error(f"Race data for '{slug}' not found. Run `python scripts/build_data.py {slug}` first.")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_shifts(slug: str) -> list:
    return load_shifts(slug)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🏎 PitMind")
    st.caption("IBM x May Challenge · F1 AI Agent")
    st.divider()

    race_labels = {slug: info["name"] for slug, info in RACES.items()}
    _race_keys = list(race_labels.keys())
    if st.session_state.selected_race not in race_labels:
        st.session_state.selected_race = _race_keys[0]
    race_slug = st.selectbox(
        "Race",
        options=_race_keys,
        format_func=lambda s: race_labels[s],
        index=_race_keys.index(st.session_state.selected_race),
        key="race_selector",
    )
    if race_slug != st.session_state.selected_race:
        st.session_state.selected_race = race_slug
        st.session_state.selected_lap = 1
        st.session_state.replay_lap = None
        st.session_state.pit_decision = None
        st.rerun()

    driver = st.selectbox(
        "Driver",
        options=DRIVERS,
        index=DRIVERS.index(st.session_state.selected_driver) if st.session_state.selected_driver in DRIVERS else 0,
        key="driver_selector",
    )
    if driver != st.session_state.selected_driver:
        st.session_state.selected_driver = driver
        st.session_state["h2h_a"] = driver  # keep H2H in sync with sidebar driver
        st.rerun()

    st.divider()

    mode = st.radio(
        "Voice",
        options=["fan", "engineer"],
        format_func=lambda m: "Fan Mode" if m == "fan" else "Engineer Mode",
        index=0 if st.session_state.mode == "fan" else 1,
        horizontal=True,
    )
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.rerun()

    st.divider()

    df_sidebar = get_race_df(st.session_state.selected_race)
    if len(df_sidebar) > 0:
        max_lap = int(df_sidebar["lap"].max())
        selected_lap = st.slider(
            "Lap",
            min_value=1,
            max_value=max_lap,
            value=st.session_state.selected_lap,
            key="lap_slider",
        )
        if selected_lap != st.session_state.selected_lap:
            st.session_state.selected_lap = selected_lap
            st.rerun()

    if st.session_state.replay_lap is not None:
        st.info(f"Replaying lap {st.session_state.replay_lap}")
        if st.button("Clear replay"):
            st.session_state.replay_lap = None
            st.session_state.pit_decision = None
            st.rerun()

    # Sync state to URL for shareable links
    st.query_params.update({
        "race": st.session_state.selected_race,
        "driver": st.session_state.selected_driver,
        "lap": str(st.session_state.selected_lap),
    })

    st.divider()
    st.caption("Built with FastF1 · OpenF1 · IBM Granite · scikit-learn · XGBoost · SHAP")

    st.sidebar.divider()
    st.sidebar.caption("**PitMind Coverage**")
    st.sidebar.caption("🏎 5 races · 8 drivers · 2025 season")
    st.sidebar.caption("🤖 IBM Granite AI · XGBoost · SHAP")
    st.sidebar.caption("📡 FastF1 + OpenF1 · Whisper sentiment")

    st.sidebar.divider()
    with st.sidebar.expander("🤖 Granite API Status", expanded=False):
        if st.button("Check connection", key="granite_ping"):
            from agent.granite import granite_health_check
            ok, msg = granite_health_check()
            if ok:
                st.success(f"✅ Connected · {msg}")
            else:
                st.error(f"❌ Unavailable · {msg}")
        st.caption("IBM Watsonx · granite-4-h-small")
        st.caption("Responses cached to reduce API calls")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

df = get_race_df(st.session_state.selected_race)
shifts = get_shifts(st.session_state.selected_race)

if len(df) == 0:
    st.stop()

tab_momentum, tab_soul, tab_pitwall, tab_track, tab_stats = st.tabs([
    "🌊 Momentum Map",
    "🧬 Driver Soul",
    "🔧 Pit Wall Mirror",
    "🏁 Track Intel",
    "📊 Driver Stats",
])

with tab_momentum:
    from app.momentum_view import render_momentum
    render_momentum(df, shifts, st.session_state.selected_driver, st.session_state.mode)

with tab_soul:
    from app.driver_view import render_driver_soul
    render_driver_soul(df, st.session_state.selected_driver, st.session_state.selected_lap, st.session_state.mode)

with tab_pitwall:
    from app.pitwall_view import render_pitwall
    render_pitwall(
        df,
        st.session_state.selected_driver,
        st.session_state.replay_lap or st.session_state.selected_lap,
        st.session_state.mode,
    )

with tab_track:
    from app.track_view import render_track_intel
    all_race_dfs = {slug: get_race_df(slug) for slug in RACES}
    render_track_intel(
        df,
        st.session_state.selected_race,
        all_race_dfs,
        st.session_state.selected_driver,
        st.session_state.selected_lap,
        st.session_state.mode,
    )

with tab_stats:
    from app.stats_view import render_driver_stats
    all_race_dfs_stats = {slug: get_race_df(slug) for slug in RACES}
    render_driver_stats(all_race_dfs_stats, st.session_state.selected_driver, st.session_state.mode)

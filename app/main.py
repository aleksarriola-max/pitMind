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

def _init_state():
    defaults = {
        "selected_race": "bahrain_2023",
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
    race_slug = st.selectbox(
        "Race",
        options=list(race_labels.keys()),
        format_func=lambda s: race_labels[s],
        index=list(race_labels.keys()).index(st.session_state.selected_race),
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
        index=DRIVERS.index(st.session_state.selected_driver),
        key="driver_selector",
    )
    if driver != st.session_state.selected_driver:
        st.session_state.selected_driver = driver
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

    st.divider()
    st.caption("Built with FastF1 · OpenF1 · IBM Granite · scikit-learn · XGBoost · SHAP")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

df = get_race_df(st.session_state.selected_race)
shifts = get_shifts(st.session_state.selected_race)

if len(df) == 0:
    st.stop()

tab_momentum, tab_soul, tab_pitwall = st.tabs([
    "🌊 Momentum Map",
    "🧬 Driver Soul",
    "🔧 Pit Wall Mirror",
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

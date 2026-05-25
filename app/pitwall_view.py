"""Pit Wall Mirror — decision interface, outcome simulation, reveal."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from agent.granite import pitwall_brief, reveal_outcome
from models.driver_soul import get_lap_state, TRAIT_COLS


@st.cache_data(show_spinner=False)
def _get_pitwall_brief(lap_data_json: str, mode: str) -> str:
    import json
    return pitwall_brief(json.loads(lap_data_json), mode)


def _simulate_outcome(df: pd.DataFrame, driver: str, decision_lap: int, choice: str) -> dict:
    """
    Fast-forward the race from decision_lap and compute the outcome difference
    between pitting and staying out.

    Returns:
        {actual_pitted: bool, actual_lap_gain/loss: int,
         simulated_position_after_5_laps: int, actual_outcome_str: str}
    """
    drv = df[df["driver"] == driver].sort_values("lap")
    rows_after = drv[drv["lap"] > decision_lap].head(10)
    rows_before = drv[drv["lap"] <= decision_lap].tail(5)

    # Did the driver actually pit?
    actual_pitted = bool(drv[(drv["lap"] > decision_lap) & (drv["lap"] <= decision_lap + 3)]["is_pit_in"].any())

    pos_at_decision = int(drv[drv["lap"] == decision_lap]["position"].iloc[0]) if len(drv[drv["lap"] == decision_lap]) > 0 else 10
    pos_5_later = pos_at_decision
    if len(rows_after) >= 5:
        pos_5_later = int(rows_after.iloc[4]["position"]) if pd.notna(rows_after.iloc[4]["position"]) else pos_at_decision

    pos_change = pos_at_decision - pos_5_later  # positive = gained positions

    actual_action = "pitted" if actual_pitted else "stayed out"
    user_correct = (
        (choice == "pit" and actual_pitted) or
        (choice == "stay" and not actual_pitted)
    )

    if pos_change > 0:
        outcome_str = f"{driver} {actual_action} and gained {pos_change} position(s) over the next 5 laps."
    elif pos_change < 0:
        outcome_str = f"{driver} {actual_action} and lost {abs(pos_change)} position(s) over the next 5 laps."
    else:
        outcome_str = f"{driver} {actual_action} and held position {pos_at_decision} for the next 5 laps."

    # Momentum trajectory for visualization
    momentum_before = drv[drv["lap"] <= decision_lap]["momentum_score"].tail(8).tolist()
    momentum_after = rows_after["momentum_score"].head(8).tolist()

    return {
        "actual_pitted": actual_pitted,
        "actual_action": actual_action,
        "user_correct": user_correct,
        "pos_at_decision": pos_at_decision,
        "pos_5_later": pos_5_later,
        "pos_change": pos_change,
        "outcome_str": outcome_str,
        "momentum_before": momentum_before,
        "momentum_after": momentum_after,
    }


def _momentum_reveal_chart(before: list, after: list, decision_lap: int, driver: str):
    laps_before = list(range(decision_lap - len(before) + 1, decision_lap + 1))
    laps_after = list(range(decision_lap + 1, decision_lap + len(after) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=laps_before, y=before,
        name="Before decision",
        line=dict(color="#888888", width=2),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=[decision_lap] + laps_after,
        y=([before[-1]] if before else [50]) + after,
        name="After decision",
        line=dict(color="#E10600", width=2.5, dash="dash"),
        mode="lines",
    ))
    fig.add_vline(
        x=decision_lap, line=dict(color="white", dash="dot", width=1),
        annotation_text="Decision lap", annotation_position="top left",
        annotation_font=dict(color="white", size=10),
    )
    fig.update_layout(
        paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
        font=dict(color="white"),
        xaxis=dict(title="Lap", gridcolor="#2A2A2A"),
        yaxis=dict(title="Momentum Score", range=[0, 100], gridcolor="#2A2A2A"),
        height=220,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    return fig


def render_pitwall(df: pd.DataFrame, driver: str, lap: int, mode: str):
    st.subheader(f"Pit Wall Mirror — {driver} · Lap {lap}")

    if len(df) == 0:
        st.info("No race data loaded.")
        return

    drv_df = df[df["driver"] == driver].sort_values("lap")
    row = drv_df[drv_df["lap"] == lap]

    if len(row) == 0:
        st.info(f"No data for {driver} at lap {lap}.")
        return

    row = row.iloc[0]

    # --- Engineer dashboard ---
    st.markdown("### What your engineers see right now")
    c1, c2, c3, c4 = st.columns(4)
    compound = row.get("tyre_compound", "?")
    tyre_age = int(row.get("tyre_age", 0))
    gap_ahead = row.get("gap_ahead", np.nan)
    gap_behind = row.get("gap_behind", np.nan)
    pace_delta = row.get("pace_delta", 0)
    pressure = row.get("pit_window_pressure", 50)
    weather = row.get("weather_track_temp", np.nan)

    compound_emoji = {"SOFT": "🔴", "MEDIUM": "🟡", "HARD": "⚪"}.get(compound, "🔵")
    c1.metric(f"{compound_emoji} Tyre", f"{compound} · {tyre_age} laps")
    c2.metric("Gap Ahead", f"{gap_ahead:.2f}s" if pd.notna(gap_ahead) else "—")
    c3.metric("Gap Behind", f"{gap_behind:.2f}s" if pd.notna(gap_behind) else "—")
    c4.metric("Pace Delta", f"+{pace_delta:.2f}s/lap" if pd.notna(pace_delta) else "—")

    c5, c6, c7, c8 = st.columns(4)
    pit_pressure_pct = int(pressure)
    sc_active = bool(row.get("safety_car_active", False))
    pos = int(row.get("position", 0)) if pd.notna(row.get("position")) else "?"
    c5.metric("Pit Window Pressure", f"{pit_pressure_pct}/100")
    c6.metric("Track Temp", f"{weather:.0f}°C" if pd.notna(weather) else "—")
    c7.metric("Safety Car", "🟡 Active" if sc_active else "✅ Clear")
    c8.metric("Position", f"P{pos}")

    # Granite brief
    lap_data = row.to_dict()
    lap_data["driver"] = driver
    lap_data_json = __import__("json").dumps({k: (v if not isinstance(v, (bool, np.bool_)) else bool(v))
                                               for k, v in lap_data.items()
                                               if isinstance(v, (int, float, str, bool, type(None)))})
    brief = _get_pitwall_brief(lap_data_json, mode)
    mode_label = "Fan" if mode == "fan" else "Engineer"
    st.markdown(f"**{mode_label} Analysis**")
    st.info(brief)

    st.divider()

    # --- Decision ---
    if st.session_state.get("pit_decision") is None:
        st.markdown("### Your call")
        col_pit, col_stay = st.columns(2)
        if col_pit.button("🔧 Pit Now", use_container_width=True, type="primary"):
            st.session_state.pit_decision = "pit"
            st.rerun()
        if col_stay.button("🏎 Stay Out", use_container_width=True):
            st.session_state.pit_decision = "stay"
            st.rerun()
    else:
        choice = st.session_state.pit_decision
        st.markdown(f"### You chose: **{'Pit Now' if choice == 'pit' else 'Stay Out'}**")

        outcome = _simulate_outcome(df, driver, lap, choice)

        # Reveal chart
        st.plotly_chart(
            _momentum_reveal_chart(
                outcome["momentum_before"],
                outcome["momentum_after"],
                lap, driver,
            ),
            use_container_width=True,
        )

        # Outcome text
        verdict_color = "green" if outcome["user_correct"] else "red"
        verdict_text = "✅ You called it!" if outcome["user_correct"] else "❌ The team disagreed."
        st.markdown(f"**{verdict_text}**")
        st.write(outcome["outcome_str"])

        # Driver Soul instinct card
        st.divider()
        st.markdown("### Driver Soul Instinct")
        lap_state = get_lap_state(df, driver, lap)
        pit_prob = lap_state.get("pit_prob", 0.5)
        instinct = "pit" if pit_prob > 0.5 else "stay out"

        top_trait = max(
            {k: lap_state.get(k, 50) for k in TRAIT_COLS}.items(),
            key=lambda x: x[1],
            default=("aggression_level", 50),
        )
        top_trait_label = top_trait[0].replace("_", " ").title()

        col_soul1, col_soul2 = st.columns([1, 2])
        col_soul1.metric(
            f"{driver} instinct model",
            f"{pit_prob:.0%} to pit" if instinct == "pit" else f"{1-pit_prob:.0%} to stay",
        )
        col_soul2.write(f"Dominant trait: **{top_trait_label}** ({top_trait[1]:.0f}/100)")

        # Granite reveal narrative
        driver_soul_ctx = {
            "pit_prob": pit_prob,
            "traits": {k: lap_state.get(k, 50) for k in TRAIT_COLS},
        }
        reveal_text = reveal_outcome(choice, outcome["outcome_str"], driver_soul_ctx, driver, mode)
        st.info(reveal_text)

        if st.button("🔄 Reset decision"):
            st.session_state.pit_decision = None
            st.rerun()

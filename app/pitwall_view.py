"""Pit Wall Mirror — decision interface, outcome simulation, reveal."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from agent.granite import pitwall_brief, reveal_outcome
from models.driver_soul import get_lap_state, TRAIT_COLS
from models.race_forecast import compare_pit_scenarios
from agent.granite import strategy_recommendation


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

    # Extract raw values (used in both modes and Granite brief)
    compound = row.get("tyre_compound", "?")
    tyre_age = int(row.get("tyre_age", 0))
    gap_ahead = row.get("gap_ahead", np.nan)
    gap_behind = row.get("gap_behind", np.nan)
    pace_delta = row.get("pace_delta", 0)
    pressure = row.get("pit_window_pressure", 50)
    weather = row.get("weather_track_temp", np.nan)
    compound_emoji = {"SOFT": "🔴", "MEDIUM": "🟡", "HARD": "⚪"}.get(compound, "🔵")
    sc_active = bool(row.get("safety_car_active", False))
    pos = int(row.get("position", 0)) if pd.notna(row.get("position")) else "?"
    pit_pressure_pct = int(pressure)

    if mode == "fan":
        st.markdown("### What's happening right now")
        tyre_health = ("Fresh" if tyre_age < 10 else "Good" if tyre_age < 20
                       else "Getting worn" if tyre_age < 30 else "Very worn — watch out")
        urgency = ("High — team wants to pit" if pressure >= 75
                   else "Moderate — watching closely" if pressure >= 50 else "Low — tyres holding")
        if pd.notna(gap_ahead):
            gap_label = ("DRS range — under attack!" if gap_ahead < 1.0
                         else "Close — pressure building" if gap_ahead < 2.5 else "Clear ahead")
        else:
            gap_label = "Leading the race"
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric(f"{compound_emoji} Tyres", compound, tyre_health)
        fc2.metric("Gap to car ahead", gap_label)
        fc3.metric("Pit urgency", urgency)
        fc4, fc5 = st.columns(2)
        fc4.metric("Race position", f"P{pos}")
        fc5.metric("Safety Car", "🟡 On track" if sc_active else "✅ Clear")
    else:
        st.markdown("### What your engineers see right now")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{compound_emoji} Tyre", f"{compound} · {tyre_age} laps")
        c2.metric("Gap Ahead", f"{gap_ahead:.2f}s" if pd.notna(gap_ahead) else "—")
        c3.metric("Gap Behind", f"{gap_behind:.2f}s" if pd.notna(gap_behind) else "—")
        c4.metric("Pace Delta", f"+{pace_delta:.2f}s/lap" if pd.notna(pace_delta) else "—")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Pit Window Pressure", f"{pit_pressure_pct}/100")
        c6.metric("Track Temp", f"{weather:.0f}°C" if pd.notna(weather) else "—")
        c7.metric("Safety Car", "🟡 Active" if sc_active else "✅ Clear")
        c8.metric("Position", f"P{pos}")

        # Driver Soul Model Confidence
        st.markdown("**Driver Soul Model Confidence**")
        _pit_p  = row.get("pit_prob", np.nan)
        _gain_p = row.get("position_gain_prob", np.nan)
        _risk_p = row.get("incident_risk", np.nan)
        _ml1, _ml2, _ml3 = st.columns(3)
        _ml1.metric("Model pit probability", f"{_pit_p:.0%}"  if pd.notna(_pit_p)  else "—")
        _ml2.metric("Position gain (5L)",    f"{_gain_p:.0%}" if pd.notna(_gain_p) else "—")
        _ml3.metric("Incident risk",         f"{_risk_p:.0%}" if pd.notna(_risk_p) else "—")

        # Pit Strategy Optimizer — Scenario Comparison Engine
        st.markdown("**Pit Strategy Optimizer — Scenario Analysis**")
        try:
            _race_slug = st.session_state.get("selected_race", "bahrain_2025")
            _scenarios = compare_pit_scenarios(df, _race_slug, lap, driver, n_laps=15)
        except Exception:
            _scenarios = []

        if _scenarios:
            _sc_cols = st.columns(4)
            for _i, _sc in enumerate(_scenarios):
                with _sc_cols[_i]:
                    _delta = _sc["position_delta"]
                    _delta_str = f"+{_delta}" if _delta > 0 else str(_delta)
                    _delta_color = "normal" if _delta == 0 else ("inverse" if _delta > 0 else "off")
                    if _sc["recommendation"]:
                        st.markdown(
                            '<div style="border:2px solid #22C55E;border-radius:6px;padding:4px;margin-bottom:4px;">'
                            f'<b style="color:#22C55E">★ {_sc["label"]}</b></div>',
                            unsafe_allow_html=True,
                        )
                    st.metric(
                        label=_sc["label"],
                        value=f"P{_sc['projected_position']}",
                        delta=f"{_delta_str} pos",
                        delta_color=_delta_color,
                    )
            st.info(strategy_recommendation(_scenarios, driver, lap, mode))

            # Scenario trajectory chart
            if any("_traj_laps" in _sc for _sc in _scenarios):
                _SCENARIO_COLORS = {
                    "Pit Now": "#22C55E", "Pit +3": "#F59E0B",
                    "Pit +5": "#3B82F6",  "Stay Out": "#EF4444",
                }
                _fig_traj = go.Figure()
                for _sc in _scenarios:
                    if "_traj_laps" not in _sc:
                        continue
                    _fig_traj.add_trace(go.Scatter(
                        x=_sc["_traj_laps"],
                        y=_sc["_traj_positions"],
                        mode="lines+markers",
                        name=f"{'★ ' if _sc['recommendation'] else ''}{_sc['label']}",
                        line=dict(
                            color=_SCENARIO_COLORS.get(_sc["label"], "#888"),
                            width=3 if _sc["recommendation"] else 1.5,
                            dash="solid" if _sc["recommendation"] else "dot",
                        ),
                        marker=dict(size=5),
                    ))
                _fig_traj.update_layout(
                    paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
                    font=dict(color="white"),
                    xaxis=dict(title="Lap", gridcolor="#2A2A2A"),
                    yaxis=dict(
                        title="Projected Position", autorange="reversed",
                        tickvals=list(range(1, 11)), gridcolor="#2A2A2A",
                    ),
                    height=260, margin=dict(l=50, r=20, t=20, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                )
                st.plotly_chart(_fig_traj, use_container_width=True)

        # Competitor Threat Analysis — engineer mode only
        st.markdown("**Competitor Threat Analysis**")
        _all_snap = df[df["lap"] == lap].copy()
        _drv_row = _all_snap[_all_snap["driver"] == driver]
        if len(_drv_row) > 0:
            _drv_pos = int(_drv_row["position"].iloc[0]) if pd.notna(_drv_row["position"].iloc[0]) else 99
            _behind_rows = _all_snap[_all_snap["position"] == _drv_pos + 1]
            if len(_behind_rows) > 0:
                _threat = _behind_rows.iloc[0]
                _t_driver   = str(_threat["driver"])
                _t_gap      = float(gap_behind) if pd.notna(gap_behind) else None
                _t_compound = str(_threat.get("tyre_compound", "?")) if pd.notna(_threat.get("tyre_compound")) else "?"
                _t_age      = int(_threat.get("tyre_age", 0)) if pd.notna(_threat.get("tyre_age")) else 0
                _t_pace     = float(_threat.get("pace_delta", 0)) if pd.notna(_threat.get("pace_delta")) else 0.0
                _closing_rate = float(pace_delta) - _t_pace if pd.notna(pace_delta) else 0.0
                if _t_gap is not None and _closing_rate > 0.01:
                    _laps_to_drs = max(0.0, (_t_gap - 1.0) / _closing_rate)
                    _threat_level = "🔴 High" if _laps_to_drs < 3 else ("🟡 Moderate" if _laps_to_drs < 8 else "🟢 Low")
                elif _t_gap is not None and _t_gap < 1.0:
                    _threat_level = "🔴 High — DRS range now"
                    _laps_to_drs = 0.0
                else:
                    _threat_level = "🟢 Low"
                    _laps_to_drs = None
                _tc1, _tc2 = st.columns([3, 1])
                _tc1.markdown(
                    f"**{_t_driver}** — {_t_compound} · {_t_age} laps"
                    + (f" · {_t_gap:.2f}s behind · closing {_closing_rate:.2f}s/lap" if _t_gap else "")
                )
                _tc2.markdown(f"**{_threat_level}**")
                if _laps_to_drs is not None and _laps_to_drs < 8:
                    st.caption(f"Estimated DRS range in ~{_laps_to_drs:.0f} lap(s)")
            else:
                st.caption("No driver directly behind — running last in group or P1")

            # Car-ahead undercut signal: is the driver in front about to pit?
            _ahead_rows = _all_snap[_all_snap["position"] == _drv_pos - 1]
            if len(_ahead_rows) > 0:
                _ah = _ahead_rows.iloc[0]
                _ah_driver   = str(_ah["driver"])
                _ah_pressure = float(_ah.get("pit_window_pressure", 50)) if pd.notna(_ah.get("pit_window_pressure")) else 50.0
                _ah_age      = int(_ah.get("tyre_age", 0)) if pd.notna(_ah.get("tyre_age")) else 0
                _ah_compound = str(_ah.get("tyre_compound", "?")) if pd.notna(_ah.get("tyre_compound")) else "?"
                _ah_hist = df[(df["driver"] == _ah_driver) & (df["lap"] >= lap - 3) & (df["lap"] <= lap)]
                if len(_ah_hist) >= 2:
                    _ah_trend = float(_ah_hist.sort_values("lap")["pit_window_pressure"].diff().mean())
                    _trend_str = (f"rising {_ah_trend:+.0f}/lap" if _ah_trend > 1
                                  else f"falling {_ah_trend:+.0f}/lap" if _ah_trend < -1
                                  else "stable")
                else:
                    _trend_str = "—"
                _urg_color = "🔴" if _ah_pressure >= 75 else ("🟡" if _ah_pressure >= 50 else "🟢")
                st.caption(
                    f"**Car ahead: {_ah_driver}** — {_ah_compound} · {_ah_age} laps · "
                    f"window pressure {_ah_pressure:.0f}/100 ({_trend_str}) {_urg_color}"
                )

        else:
            # Fallback: original argmax display
            _drv_full = df[df["driver"] == driver].sort_values("lap")
            _ahead_window = _drv_full[(_drv_full["lap"] >= lap) & (_drv_full["lap"] <= lap + 15)]
            if len(_ahead_window) > 0 and "pit_window_pressure" in _ahead_window.columns:
                _peak_idx = _ahead_window["pit_window_pressure"].idxmax()
                _optimal_lap = int(_ahead_window.loc[_peak_idx, "lap"])
                _peak_pressure = int(_ahead_window.loc[_peak_idx, "pit_window_pressure"])
                _laps_to_optimal = _optimal_lap - lap
                if _laps_to_optimal == 0:
                    st.warning(f"**Pit Strategy Model:** Pit **now** — window pressure peaks this lap ({_peak_pressure}/100)")
                else:
                    st.info(f"**Pit Strategy Model:** Optimal pit in **{_laps_to_optimal} lap(s)** (lap {_optimal_lap}) · pressure peaks at {_peak_pressure}/100")

        # Compound What-If
        st.markdown("**Strategy Scenarios — If You Pit Now**")

        # Compute data-derived pace boost per compound from fresh-tyre laps
        _fresh = df[(df["tyre_age"] <= 5) & (df["pace_delta"].notna())]
        _pace_by_compound = _fresh.groupby("tyre_compound")["pace_delta"].median().to_dict()
        _current_compound_pace = _pace_by_compound.get(compound, pace_delta)
        def _boost(cmp):
            base = _pace_by_compound.get(cmp, 0.0)
            return float(base - _current_compound_pace)

        COMPOUND_PARAMS = {
            "SOFT":   {"color": "🔴", "stint_laps": 18, "pace_boost": _boost("SOFT"),   "label": "Soft"},
            "MEDIUM": {"color": "🟡", "stint_laps": 28, "pace_boost": _boost("MEDIUM"), "label": "Medium"},
            "HARD":   {"color": "⚪", "stint_laps": 40, "pace_boost": _boost("HARD"),   "label": "Hard"},
        }
        wf_cols = st.columns(3)
        for i, (cmp, params) in enumerate(COMPOUND_PARAMS.items()):
            with wf_cols[i]:
                # Estimate re-entry pace delta: fresh tyres = boost vs current degraded pace
                fresh_delta = pace_delta + params["pace_boost"]  # negative = faster than field
                # Estimate if we undercut the car ahead
                if pd.notna(gap_ahead) and pd.notna(pace_delta):
                    pace_gain_per_lap = pace_delta - fresh_delta   # positive = fresh tyres are faster
                    if pace_gain_per_lap > 0.01:
                        laps_to_close = gap_ahead / pace_gain_per_lap
                        can_undercut = laps_to_close < 8
                    else:
                        can_undercut = False
                else:
                    can_undercut = False
                # Estimate if car behind will undercut us
                if pd.notna(gap_behind):
                    at_risk = gap_behind < 3.0
                else:
                    at_risk = False

                if pd.notna(gap_behind) and pd.notna(pace_delta) and pace_delta > 0:
                    pace_diff = pace_delta - fresh_delta
                    undercut_prob = min(100, max(0, int((pace_diff / max(gap_behind, 0.1)) * 100)))
                else:
                    undercut_prob = 0

                verdict = "✅ Undercut" if can_undercut else ("⚠️ Risky" if at_risk else "↔ Neutral")
                st.metric(
                    f"{params['color']} {params['label']}",
                    verdict,
                    f"~{params['stint_laps']}L · {fresh_delta:+.2f}s · {undercut_prob}% undercut",
                )

        # Proactive alerts
        alerts = []
        if pressure >= 80:
            alerts.append(("🚨 Critical pit window", f"Pressure at {pit_pressure_pct}/100 — team is calling it now", "error"))
        elif pressure >= 65:
            alerts.append(("⚠️ Pit window opening", f"Pressure at {pit_pressure_pct}/100 — monitor closely", "warning"))
        if tyre_age >= 35:
            alerts.append(("🔥 Tyre life critical", f"{tyre_age} laps on {compound} — pace drop imminent", "error"))
        elif tyre_age >= 25:
            alerts.append(("⚠️ Tyre wear elevated", f"{tyre_age} laps on {compound}", "warning"))
        if sc_active:
            alerts.append(("🟡 Safety Car opportunity", "Free pit stop window — pit cost near zero", "warning"))
        if pd.notna(gap_ahead) and gap_ahead < 1.0:
            alerts.append(("⚡ DRS range", f"Only {gap_ahead:.2f}s to car ahead — overtake or defend", "warning"))
        for title, msg, level in alerts:
            if level == "error":
                st.error(f"**{title}** — {msg}")
            else:
                st.warning(f"**{title}** — {msg}")

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

        # Pit Accuracy Overlay (engineer only)
        if mode == "engineer":
            import plotly.graph_objects as _go
            drv_all = df[df["driver"] == driver].sort_values("lap")
            pw = drv_all[["lap", "pit_window_pressure"]].dropna()
            if len(pw) > 0:
                st.markdown("**Pit Window Pressure vs. Actual Pit Laps**")
                actual_pit_laps = (
                    drv_all[drv_all["is_pit_in"] == True]["lap"].tolist()
                    if "is_pit_in" in drv_all.columns else []
                )
                fig_pw = _go.Figure()
                fig_pw.add_trace(_go.Scatter(
                    x=pw["lap"], y=pw["pit_window_pressure"],
                    mode="lines", line=dict(color="#F59E0B", width=2),
                    fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                    name="Pit Window Pressure",
                    hovertemplate="Lap %{x}<br>Pressure: %{y}/100<extra></extra>",
                ))
                for pit_lap in actual_pit_laps:
                    fig_pw.add_vline(
                        x=pit_lap, line=dict(color="#22C55E", dash="solid", width=2),
                        annotation_text=f"Pit L{int(pit_lap)}",
                        annotation_position="top left",
                        annotation_font=dict(color="#22C55E", size=9),
                    )
                fig_pw.update_layout(
                    paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
                    font=dict(color="white"),
                    xaxis=dict(title="Lap", gridcolor="#2A2A2A"),
                    yaxis=dict(title="Pit Window Pressure (0–100)", range=[0, 105], gridcolor="#2A2A2A"),
                    height=200, margin=dict(l=60, r=20, t=10, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_pw, use_container_width=True)

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

    # Engineer chat interface
    if mode == "engineer":
        st.divider()
        st.markdown("### Ask the Pit Wall")
        st.caption("Ask anything about this lap's telemetry, strategy, or driver state.")

        if "pitwall_chat_history" not in st.session_state:
            st.session_state.pitwall_chat_history = []

        # Display existing chat history
        for msg in st.session_state.pitwall_chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Build lap context dict for Granite
        lap_ctx = {
            "compound": row.get("tyre_compound", "?"),
            "tyre_age_laps": int(row.get("tyre_age", 0)),
            "gap_ahead_s": f"{gap_ahead:.2f}" if pd.notna(gap_ahead) else "leading",
            "gap_behind_s": f"{gap_behind:.2f}" if pd.notna(gap_behind) else "N/A",
            "pace_delta_s_per_lap": f"{pace_delta:+.2f}" if pd.notna(pace_delta) else "N/A",
            "pit_window_pressure": int(pressure),
            "position": pos,
            "safety_car": sc_active,
            "track_temp_c": f"{weather:.0f}" if pd.notna(weather) else "N/A",
        }

        # Suggested questions (shown when chat is empty)
        if not st.session_state.pitwall_chat_history:
            st.caption("Quick questions:")
            _suggestions = [
                "Should I pit now?",
                "What's my undercut risk?",
                "How are my tyres holding up vs the field?",
            ]
            _sq_cols = st.columns(3)
            for _i, (_col, _sug) in enumerate(zip(_sq_cols, _suggestions)):
                if _col.button(_sug, key=f"_sq_{_i}"):
                    st.session_state["_pitwall_prefill"] = _sug
                    st.rerun()

        question = st.chat_input("e.g. Should I pit now? How are the tyres holding up?")
        # Handle prefilled question from suggestion button
        if not question and st.session_state.get("_pitwall_prefill"):
            question = st.session_state.pop("_pitwall_prefill")
        if question:
            st.session_state.pitwall_chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Granite thinking..."):
                    from agent.granite import pitwall_chat
                    answer = pitwall_chat(question, lap_ctx, driver, mode)
                st.write(answer)
            st.session_state.pitwall_chat_history.append({"role": "assistant", "content": answer})

        if st.session_state.pitwall_chat_history:
            if st.button("Clear chat", key="clear_pitwall_chat"):
                st.session_state.pitwall_chat_history = []
                st.rerun()

"""Momentum Map — lap-by-lap chart with flag overlays, shift markers, and race narrative."""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data.ingest import DRIVERS

try:
    from agent.granite import annotate_shift, race_narrative, driver_of_race
except ImportError:
    def annotate_shift(shift, mode="fan"):
        d = shift.get("driver", ""); lap = shift.get("lap", 0); mag = shift.get("magnitude", 0)
        return f"Lap {lap}: {d} momentum shift of {mag:.0f} pts."
    def race_narrative(shifts, flags, positions, race_name, mode="fan"):
        winner = min(positions.items(), key=lambda x: x[1])[0] if positions else "?"
        return f"{winner} won {race_name}. Full narrative unavailable — Granite API not connected."
    def driver_of_race(driver, stats, mode="fan"):
        return f"{driver} was the standout performer this race."

TEAM_COLORS = {
    "Red Bull": "#3671C6",
    "Mercedes": "#27F4D2",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "AlphaTauri": "#6692FF",
    "Alfa Romeo": "#C92D4B",
    "Haas": "#B6BABD",
}

DRIVER_TEAMS = {
    "VER": "Red Bull", "PER": "Red Bull",
    "HAM": "Ferrari",  "RUS": "Mercedes",
    "LEC": "Ferrari",  "SAI": "Williams",
    "NOR": "McLaren",
    "ALO": "Aston Martin",
}

FLAG_COLORS = {
    "SAFETY_CAR":        ("rgba(200,200,200,0.15)", "⬜ SC"),
    "VIRTUAL_SAFETY_CAR": ("rgba(100,160,255,0.15)", "🔵 VSC"),
    "YELLOW":            ("rgba(255,220,0,0.10)",   "🟡 Yellow"),
    "DOUBLE_YELLOW":     ("rgba(255,180,0,0.15)",   "🟡🟡 Double Yellow"),
    "RED":               ("rgba(220,50,50,0.15)",   "🔴 Red Flag"),
}


def _derive_flag_periods(race_slug: str, df: pd.DataFrame) -> list:
    """Derive flag periods without depending on load_flags from ingest.
    Tries the pre-saved JSON first; falls back to safety_car_active column."""
    import os, json
    cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "cache", f"{race_slug}_flags.json")
    try:
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)
    except Exception:
        pass

    # Fallback: derive SC periods from safety_car_active column
    if "safety_car_active" not in df.columns:
        return []
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


@st.cache_data(show_spinner=False)
def _get_shift_annotation(shift_json: str, mode: str) -> str:
    shift = json.loads(shift_json)
    return annotate_shift(shift, mode)


@st.cache_data(show_spinner=False)
def _get_race_narrative(race_slug: str, shifts_json: str, flags_json: str, positions_json: str, race_name: str, mode: str) -> str:
    shifts = json.loads(shifts_json)
    flags = json.loads(flags_json)
    positions = json.loads(positions_json)
    return race_narrative(shifts, flags, positions, race_name, mode)


@st.cache_data(show_spinner=False)
def _get_driver_of_race(race_slug: str, driver: str, stats_json: str, mode: str) -> str:
    stats = json.loads(stats_json)
    return driver_of_race(driver, stats, mode)


def _pick_driver_of_race(df: pd.DataFrame, shifts: list) -> tuple[str, dict]:
    """Pick the standout driver by positions gained + net momentum gain + fewest errors."""
    if len(df) == 0:
        return "N/A", {}
    candidates = {}
    for driver, grp in df.groupby("driver"):
        grp = grp.sort_values("lap")
        grid = grp["grid_position"].dropna() if "grid_position" in grp.columns else pd.Series(dtype=float)
        final_pos = grp["position"].dropna() if "position" in grp.columns else pd.Series(dtype=float)
        gained = (int(grid.iloc[0]) - int(final_pos.iloc[-1])) if len(grid) > 0 and len(final_pos) > 0 else 0
        driver_shifts = [s for s in shifts if s.get("driver") == driver]
        momentum_gain = sum(s["magnitude"] for s in driver_shifts if s.get("direction") == "up") - \
                        sum(s["magnitude"] for s in driver_shifts if s.get("direction") == "down")
        score = gained * 10 + momentum_gain
        traits = {c: float(grp[c].dropna().mean()) for c in [
            "overtake_tendency", "aggression_level", "tyre_preservation",
            "pressure_consistency", "restart_aggression"
        ] if c in grp.columns and grp[c].notna().any()}
        top_trait = max(traits.items(), key=lambda x: x[1], default=("aggression_level", 70))
        candidates[driver] = {
            "score": score, "positions_gained": gained, "momentum_gain": momentum_gain,
            "error_count": 0, "top_trait": top_trait[0], "top_trait_val": top_trait[1],
        }
    if not candidates:
        return "N/A", {}
    best = max(candidates.items(), key=lambda x: x[1]["score"])
    return best[0], best[1]


def render_momentum(df: pd.DataFrame, shifts: list, highlight_driver: str, mode: str):
    st.subheader("Momentum Map")

    if len(df) == 0:
        st.info("No race data loaded.")
        return

    race_name = df["race"].iloc[0] if "race" in df.columns else "Race"
    race_slug = df["race"].iloc[0].lower().replace(" ", "_") if "race" in df.columns else ""
    st.caption(f"{race_name} · Momentum score per car per lap (0–100)")

    # Load flag periods — try pre-saved JSON, fall back to safety_car_active column
    flag_periods = _derive_flag_periods(race_slug, df)

    # Driver filter
    available_drivers = [d for d in DRIVERS if d in df["driver"].unique()]
    selected_drivers = st.multiselect(
        "Drivers to show",
        options=available_drivers,
        default=available_drivers,
        key="momentum_drivers",
    )

    if not selected_drivers:
        st.info("Select at least one driver.")
        return

    # ── Momentum chart ──────────────────────────────────────────────────────────
    fig = go.Figure()
    df_plot = df[df["driver"].isin(selected_drivers)].sort_values(["driver", "lap"])

    for driver in selected_drivers:
        d = df_plot[df_plot["driver"] == driver]
        team = DRIVER_TEAMS.get(driver, "")
        color = TEAM_COLORS.get(team, "#AAAAAA")
        fig.add_trace(go.Scatter(
            x=d["lap"], y=d["momentum_score"],
            name=driver, mode="lines",
            line=dict(color=color, width=3 if driver == highlight_driver else 1.5),
            opacity=1.0 if driver == highlight_driver else 0.5,
            hovertemplate=f"<b>{driver}</b><br>Lap %{{x}}<br>Momentum: %{{y:.1f}}<extra></extra>",
        ))

    # Shift markers
    top_shifts = [s for s in shifts if s["driver"] in selected_drivers and s["magnitude"] > 10][:15]
    for shift in top_shifts:
        color = "#22C55E" if shift["direction"] == "up" else "#EF4444"
        arrow = "▲" if shift["direction"] == "up" else "▼"
        fig.add_vline(x=shift["lap"], line=dict(color=color, dash="dot", width=1), opacity=0.4)
        fig.add_annotation(
            x=shift["lap"], y=shift["momentum_after"],
            text=f"{arrow} {shift['driver']} L{shift['lap']}",
            showarrow=False, font=dict(size=9, color=color), yshift=8,
        )

    # Flag overlays from flag_periods (richer) or safety_car_active fallback
    if flag_periods:
        for p in flag_periods:
            fill_color, _ = FLAG_COLORS.get(p["flag"], ("rgba(200,200,200,0.08)", ""))
            fig.add_vrect(
                x0=p["lap_start"] - 0.5, x1=p["lap_end"] + 0.5,
                fillcolor=fill_color, line_width=0,
            )
    else:
        for lap in df[df["safety_car_active"]]["lap"].unique():
            fig.add_vrect(x0=lap - 0.4, x1=lap + 0.4, fillcolor="rgba(200,200,200,0.1)", line_width=0)

    fig.update_layout(
        paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
        font=dict(color="white"),
        xaxis=dict(title="Lap", gridcolor="#2A2A2A"),
        yaxis=dict(title="Momentum Score", gridcolor="#2A2A2A", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=20, b=40),
        height=420, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Flag legend
    if flag_periods:
        present = {p["flag"] for p in flag_periods}
        legend_parts = [FLAG_COLORS[f][1] for f in ["SAFETY_CAR", "VIRTUAL_SAFETY_CAR", "YELLOW", "RED"] if f in present]
        if legend_parts:
            st.caption("Flag periods: " + " · ".join(legend_parts))

    st.divider()

    # ── Shift detail panel ──────────────────────────────────────────────────────
    st.subheader("Momentum Shifts")
    if top_shifts:
        cols = st.columns([1, 1, 3])
        cols[0].markdown("**Lap / Driver**")
        cols[1].markdown("**Shift**")
        cols[2].markdown("**Granite annotation**")
        for shift in top_shifts[:8]:
            cols = st.columns([1, 1, 3])
            icon = "🟢" if shift["direction"] == "up" else "🔴"
            cols[0].write(f"Lap {shift['lap']} · {shift['driver']}")
            cols[1].write(f"{icon} {shift['magnitude']:.0f} pts")
            cols[2].write(_get_shift_annotation(json.dumps(shift), mode))
            if cols[0].button("Replay", key=f"replay_{shift['lap']}_{shift['driver']}"):
                st.session_state.selected_lap = shift["lap"]
                st.session_state.replay_lap = shift["lap"]
                st.session_state.selected_driver = shift["driver"]
                st.session_state.pit_decision = None
                st.rerun()
    else:
        st.info("No significant momentum shifts detected for the selected drivers.")

    st.divider()

    # ── Race Narrative AI cards ─────────────────────────────────────────────────
    st.subheader("Race Intelligence")
    mode_label = "Fan" if mode == "fan" else "Engineer"

    # Card 1 — Explain This Race
    with st.expander("📖 Explain This Race", expanded=False):
        if st.button("Generate race narrative", key="gen_narrative"):
            final_positions = {}
            for driver, grp in df.groupby("driver"):
                pos = grp.sort_values("lap")["position"].dropna()
                if len(pos) > 0:
                    final_positions[driver] = int(pos.iloc[-1])
            narrative = _get_race_narrative(
                race_slug,
                json.dumps(shifts[:10]),
                json.dumps(flag_periods),
                json.dumps(final_positions),
                race_name,
                mode,
            )
            st.info(f"**{mode_label} Mode:** {narrative}")

    # Card 2 — Driver of the Race
    with st.expander("🏆 Driver of the Race", expanded=False):
        dotr_driver, dotr_stats = _pick_driver_of_race(df, shifts)
        st.markdown(f"**Model pick: {dotr_driver}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Positions Gained", f"+{dotr_stats.get('positions_gained', 0)}")
        col2.metric("Momentum Gain", f"{dotr_stats.get('momentum_gain', 0):.0f} pts")
        col3.metric("Dominant Trait", dotr_stats.get("top_trait", "—").replace("_", " ").title())
        narrative = _get_driver_of_race(race_slug, dotr_driver, json.dumps(dotr_stats), mode)
        st.info(f"**{mode_label} Analysis:** {narrative}")

    # Card 3 — Top 5 Key Moments
    with st.expander("⚡ Top 5 Key Moments", expanded=True):
        top5 = sorted(shifts, key=lambda s: s.get("magnitude", 0), reverse=True)[:5]
        if not top5:
            st.info("No momentum shift data available.")
        else:
            for i, shift in enumerate(top5):
                icon = "🟢" if shift["direction"] == "up" else "🔴"
                delta_label = f"+{shift['magnitude']:.0f}" if shift["direction"] == "up" else f"-{shift['magnitude']:.0f}"
                cols = st.columns([0.3, 0.15, 2])
                cols[0].markdown(f"**#{i+1} Lap {shift['lap']} — {shift['driver']}**")
                cols[1].markdown(f"{icon} **{delta_label} pts**")
                annotation = _get_shift_annotation(json.dumps(shift), mode)
                cols[2].markdown(annotation)
                if i < len(top5) - 1:
                    st.markdown("---")

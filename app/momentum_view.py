"""Momentum Map — flowing lap-by-lap line chart with annotated shift markers."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data.ingest import DRIVERS
from agent.granite import annotate_shift

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
    "HAM": "Mercedes", "RUS": "Mercedes",
    "LEC": "Ferrari", "SAI": "Ferrari",
    "NOR": "McLaren",
    "ALO": "Aston Martin",
}


@st.cache_data(show_spinner=False)
def _get_shift_annotation(shift_json: str, mode: str) -> str:
    import json
    shift = json.loads(shift_json)
    return annotate_shift(shift, mode)


def render_momentum(df: pd.DataFrame, shifts: list, highlight_driver: str, mode: str):
    st.subheader("Momentum Map")

    if len(df) == 0:
        st.info("No race data loaded.")
        return

    race_name = df["race"].iloc[0] if "race" in df.columns else "Race"
    st.caption(f"{race_name} · Momentum score per car per lap (0–100)")

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

    # Build Plotly figure
    fig = go.Figure()
    df_plot = df[df["driver"].isin(selected_drivers)].sort_values(["driver", "lap"])

    for driver in selected_drivers:
        d = df_plot[df_plot["driver"] == driver]
        team = DRIVER_TEAMS.get(driver, "")
        color = TEAM_COLORS.get(team, "#AAAAAA")
        line_width = 3 if driver == highlight_driver else 1.5
        opacity = 1.0 if driver == highlight_driver else 0.5

        fig.add_trace(go.Scatter(
            x=d["lap"],
            y=d["momentum_score"],
            name=driver,
            mode="lines",
            line=dict(color=color, width=line_width),
            opacity=opacity,
            hovertemplate=(
                f"<b>{driver}</b><br>"
                "Lap %{x}<br>"
                "Momentum: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        ))

    # Shift markers — top shifts only (those involving selected drivers)
    top_shifts = [
        s for s in shifts
        if s["driver"] in selected_drivers and s["magnitude"] > 10
    ][:15]

    for shift in top_shifts:
        direction_color = "#22C55E" if shift["direction"] == "up" else "#EF4444"
        arrow = "▲" if shift["direction"] == "up" else "▼"
        fig.add_vline(
            x=shift["lap"],
            line=dict(color=direction_color, dash="dot", width=1),
            opacity=0.4,
        )
        fig.add_annotation(
            x=shift["lap"],
            y=shift["momentum_after"],
            text=f"{arrow} {shift['driver']} L{shift['lap']}",
            showarrow=False,
            font=dict(size=9, color=direction_color),
            yshift=8,
        )

    # Safety car overlays
    sc_laps = df[df["safety_car_active"]]["lap"].unique()
    for lap in sc_laps:
        fig.add_vrect(
            x0=lap - 0.4, x1=lap + 0.4,
            fillcolor="yellow", opacity=0.08,
            line_width=0,
        )

    fig.update_layout(
        paper_bgcolor="#0F0F0F",
        plot_bgcolor="#0F0F0F",
        font=dict(color="white"),
        xaxis=dict(title="Lap", gridcolor="#2A2A2A", showgrid=True),
        yaxis=dict(title="Momentum Score", gridcolor="#2A2A2A", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=20, b=40),
        height=420,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Safety car legend
    if len(sc_laps) > 0:
        st.caption(f"🟡 Yellow bands = Safety Car laps ({', '.join(str(l) for l in sorted(sc_laps)[:8])}...)")

    # Shift detail panel
    st.subheader("Momentum Shifts")
    if top_shifts:
        cols = st.columns([1, 1, 3])
        cols[0].markdown("**Lap / Driver**")
        cols[1].markdown("**Shift**")
        cols[2].markdown("**Granite annotation**")

        for shift in top_shifts[:8]:
            import json
            cols = st.columns([1, 1, 3])
            direction_icon = "🟢" if shift["direction"] == "up" else "🔴"
            cols[0].write(f"Lap {shift['lap']} · {shift['driver']}")
            cols[1].write(f"{direction_icon} {shift['magnitude']:.0f} pts")

            annotation = _get_shift_annotation(json.dumps(shift), mode)
            cols[2].write(annotation)

            # Replay button
            if cols[0].button(f"Replay", key=f"replay_{shift['lap']}_{shift['driver']}"):
                st.session_state.selected_lap = shift["lap"]
                st.session_state.replay_lap = shift["lap"]
                st.session_state.selected_driver = shift["driver"]
                st.session_state.pit_decision = None
                st.rerun()
    else:
        st.info("No significant momentum shifts detected for the selected drivers.")

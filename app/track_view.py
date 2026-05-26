"""Track Intel — circuit profile, sector dominance, grid analysis, battle prediction. v2."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from models.race_forecast import forecast_positions, get_aggression_zone_stats

try:
    from agent.granite import track_intel_brief, error_summary
except ImportError:
    def track_intel_brief(circuit, sectors, mode="fan"):
        return f"Track Intel: {circuit.get('track_character','')[:80]}. Granite narrative unavailable."
    def error_summary(errors, driver, mode="fan"):
        return f"{driver} incident summary unavailable — Granite not connected."

try:
    from models.error_detection import detect_pilot_errors, error_counts, total_errors
except ImportError:
    def detect_pilot_errors(df, flag_periods=None):
        import pandas as pd; return pd.DataFrame(columns=["lap","driver","error_type","severity","signal_value","description"])
    def error_counts(df): return {}
    def total_errors(df, driver): return 0

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
}

CIRCUIT_PROFILES = {
    "bahrain_2025": {
        "length_km": 5.412, "laps": 57, "drs_zones": 3,
        "overtaking_difficulty": 28,
        "key_corners": ["Turn 1 braking", "Turn 4 hairpin", "Turn 10 chicane"],
        "aggression_sectors": [1, 3],
        "track_character": "Power circuit with long straights and abrasive surface. High tyre degradation rewards early aggressive strategy. DRS effective on the main straight and back section.",
    },
    "monaco_2025": {
        "length_km": 3.337, "laps": 78, "drs_zones": 1,
        "overtaking_difficulty": 96,
        "key_corners": ["Sainte Devote", "Grand Hotel Hairpin", "Swimming Pool"],
        "aggression_sectors": [2],
        "track_character": "Narrow street circuit where qualifying is everything. Track position is impossible to recover once lost. The only overtake opportunity is the pit window.",
    },
    "silverstone_2025": {
        "length_km": 5.891, "laps": 52, "drs_zones": 2,
        "overtaking_difficulty": 42,
        "key_corners": ["Copse", "Maggotts-Becketts", "Chapel", "Stowe"],
        "aggression_sectors": [1, 2],
        "track_character": "High-speed flowing circuit demanding aerodynamic balance. Fast corners reward confidence; heavy braking zones at Stowe and Club create overtake opportunities.",
    },
    "monza_2025": {
        "length_km": 5.793, "laps": 53, "drs_zones": 2,
        "overtaking_difficulty": 22,
        "key_corners": ["Turn 1 braking", "Ascari chicane", "Parabolica"],
        "aggression_sectors": [1, 3],
        "track_character": "Temple of Speed — lowest downforce circuit. Massive braking zones at Turn 1 and Ascari create the most overtaking opportunities of the season. Slipstream battles define lap 1.",
    },
    "abudhabi_2025": {
        "length_km": 5.281, "laps": 58, "drs_zones": 3,
        "overtaking_difficulty": 38,
        "key_corners": ["Turn 5 hairpin", "Turn 8 complex", "Turn 21 final"],
        "aggression_sectors": [1, 3],
        "track_character": "Season finale circuit with a mix of high-speed and technical sections. Three DRS zones make it one of the better circuits for overtaking. Tyre management critical in the desert heat.",
    },
}

DRIVER_COLORS = {
    "VER": "#3671C6", "HAM": "#E8002D", "LEC": "#E8002D",
    "SAI": "#64C4FF", "ALO": "#229971", "PER": "#3671C6",
    "NOR": "#FF8000", "RUS": "#27F4D2",
}

# Hand-crafted circuit path waypoints (x, y normalized).
# Each path is closed (first == last point), traced in racing direction.
CIRCUIT_PATHS = {
    "bahrain": {
        "name": "Bahrain International Circuit",
        # Clockwise. S/F at left, main straight going right.
        # T1-3 complex at top-right, T10 hairpin at bottom-left.
        "x": [1.0, 7.0, 7.8, 8.5, 9.0, 9.0, 8.7, 8.2, 7.8, 7.3, 7.0, 6.5,
              6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.7, 1.5, 1.7,
              2.2, 2.8, 3.3, 3.5, 3.2, 2.8, 2.2, 1.5, 1.0, 1.0],
        "y": [5.5, 5.5, 5.9, 5.5, 5.0, 4.2, 3.5, 3.0, 2.7, 2.5, 2.3, 2.1,
              1.9, 1.7, 1.5, 1.3, 1.1, 1.0, 0.9, 0.9, 1.0, 1.3, 1.8, 2.3,
              2.8, 3.3, 3.8, 4.3, 4.8, 5.2, 5.5, 5.6, 5.6, 5.5],
    },
    "monaco": {
        "name": "Circuit de Monaco",
        # Clockwise. S/F bottom-left, Casino at top, Grand Hotel Hairpin on right.
        # Tunnel goes through lower-right section.
        "x": [1.5, 5.5, 6.0, 6.7, 7.5, 8.3, 8.8, 9.2, 9.5, 9.5, 9.3, 9.0,
              9.0, 9.3, 9.5, 9.3, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5,
              5.0, 4.5, 4.2, 4.5, 4.7, 4.5, 4.0, 3.5, 3.0, 2.5, 2.2, 1.8,
              1.5, 1.5],
        "y": [3.5, 3.5, 4.0, 5.0, 6.0, 7.0, 7.3, 7.0, 6.5, 6.0, 5.5, 5.0,
              4.5, 4.0, 3.5, 3.0, 3.0, 3.2, 3.5, 3.3, 3.0, 2.8, 2.7, 2.8,
              3.0, 3.2, 3.0, 2.7, 2.4, 2.1, 2.4, 2.7, 2.9, 3.1, 3.3, 3.4,
              3.5, 3.5],
    },
    "silverstone": {
        "name": "Silverstone Circuit",
        # Clockwise. S/F at left, Maggotts-Becketts esses at top-right.
        "x": [1.0, 7.0, 7.8, 8.5, 8.8, 8.5, 7.8, 7.3, 7.5, 8.0, 8.3, 8.0,
              8.5, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.0, 3.0, 2.0,
              1.5, 1.0, 0.8, 1.0, 1.0],
        "y": [5.0, 5.0, 5.3, 5.8, 6.5, 7.0, 6.5, 6.2, 5.8, 5.5, 5.2, 4.8,
              4.3, 3.5, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.8, 0.8, 1.0,
              1.5, 2.5, 3.5, 4.5, 5.0],
    },
    "monza": {
        "name": "Autodromo Nazionale Monza",
        # Clockwise. S/F top-left. Oval with chicanes.
        # Variante Rettifilo (top-right), Curve Grande, Roggia, Lesmos, Ascari, Parabolica.
        "x": [2.0, 5.5, 6.3, 7.0, 6.7, 7.2, 7.8, 8.5, 9.0, 9.5, 9.5, 9.5,
              9.2, 8.8, 8.5, 8.8, 9.0, 8.7, 8.3, 8.0, 7.5, 7.0, 6.5, 6.0,
              5.5, 5.0, 4.5, 4.3, 4.5, 4.3, 3.8, 3.2, 2.5, 1.8, 1.3, 1.0,
              1.5, 2.0, 2.0],
        "y": [6.5, 6.5, 6.8, 6.5, 6.2, 6.5, 6.2, 5.8, 5.3, 4.8, 4.2, 3.6,
              3.2, 3.0, 2.7, 2.5, 2.3, 2.0, 1.8, 2.0, 1.8, 1.5, 1.2, 1.0,
              1.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.8, 5.5, 6.0, 6.3,
              6.5, 6.5, 6.5],
    },
    "abudhabi": {
        "name": "Yas Marina Circuit",
        # Anti-clockwise. S/F at top-right, T1 hairpin at far left.
        # Hotel section creates distinctive double-loop on the right side.
        "x": [9.0, 2.0, 1.5, 1.5, 2.0, 3.0, 4.0, 5.0, 5.5, 5.5, 5.0, 4.5,
              4.0, 3.5, 3.2, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5,
              9.8, 9.5, 9.0, 8.5, 9.0, 9.5, 9.8, 9.5, 9.0, 8.5, 8.0, 9.0,
              9.0],
        "y": [5.5, 5.5, 5.0, 4.5, 4.0, 3.5, 3.2, 3.0, 3.3, 2.8, 2.5, 2.2,
              2.0, 1.7, 1.4, 1.1, 0.9, 0.8, 0.9, 1.1, 1.5, 2.0, 2.8, 3.5,
              4.2, 4.8, 5.0, 5.2, 5.5, 5.8, 6.2, 6.5, 6.3, 6.0, 5.8, 5.8,
              5.5],
    },
}
CIRCUIT_PATHS["abu_dhabi"] = CIRCUIT_PATHS["abudhabi"]


def _position_on_path(x_s: list, y_s: list, fraction: float) -> tuple:
    """Return the (x, y) coordinate at `fraction` (0.0–1.0) along the arc-length of the path."""
    xs, ys = np.array(x_s), np.array(y_s)
    dx, dy = np.diff(xs), np.diff(ys)
    seg_lengths = np.sqrt(dx**2 + dy**2)
    cumulative = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    target = (fraction % 1.0) * total
    idx = int(np.searchsorted(cumulative, target) - 1)
    idx = max(0, min(idx, len(xs) - 2))
    remaining = target - cumulative[idx]
    seg_len = seg_lengths[idx]
    t = remaining / seg_len if seg_len > 0 else 0.0
    return float(xs[idx] + t * dx[idx]), float(ys[idx] + t * dy[idx])


def _render_circuit_map(race_slug: str, lap_df=None):
    """Return a Plotly figure with the circuit layout and optional driver position dots."""
    path_data = None
    for key in CIRCUIT_PATHS:
        if key in race_slug.lower() and CIRCUIT_PATHS[key] is not None:
            path_data = CIRCUIT_PATHS[key]
            break
    if not path_data:
        return None

    x_pts = list(path_data["x"])
    y_pts = list(path_data["y"])

    # Smooth path with cubic spline if scipy is available
    try:
        from scipy.interpolate import CubicSpline
        t = np.linspace(0, 1, len(x_pts))
        t_new = np.linspace(0, 1, 400)
        x_s = CubicSpline(t, x_pts, bc_type="periodic" if x_pts[0] == x_pts[-1] else "not-a-knot")(t_new).tolist()
        y_s = CubicSpline(t, y_pts, bc_type="periodic" if y_pts[0] == y_pts[-1] else "not-a-knot")(t_new).tolist()
    except Exception:
        x_s, y_s = x_pts, y_pts

    # Axis ranges with padding
    pad = 0.4
    x_min, x_max = min(x_s) - pad, max(x_s) + pad
    y_min, y_max = min(y_s) - pad, max(y_s) + pad

    fig = go.Figure()

    # Track outline
    fig.add_trace(go.Scatter(
        x=x_s, y=y_s,
        mode="lines",
        line=dict(color="#E8002D", width=5),
        showlegend=False,
        hoverinfo="skip",
    ))

    # S/F marker
    fig.add_trace(go.Scatter(
        x=[x_pts[0]], y=[y_pts[0]],
        mode="markers+text",
        marker=dict(size=10, color="white", symbol="square"),
        text=["S/F"],
        textposition="top right",
        textfont=dict(color="white", size=11),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Driver position dots for this lap
    if lap_df is not None and len(lap_df) > 0 and "position" in lap_df.columns:
        lap_sorted = lap_df.dropna(subset=["position"]).sort_values("position").head(8)
        lap_time_est = float(lap_df["lap_time"].median()) if "lap_time" in lap_df.columns else 90.0
        if pd.isna(lap_time_est) or lap_time_est <= 0:
            lap_time_est = 90.0
        LEADER_FRACTION = 0.92
        cumulative_gap = 0.0
        for _, row in lap_sorted.iterrows():
            gap = float(row["gap_ahead"]) if "gap_ahead" in row and pd.notna(row["gap_ahead"]) else 0.0
            cumulative_gap += gap
            fraction = (LEADER_FRACTION - cumulative_gap / lap_time_est) % 1.0
            x_d, y_d = _position_on_path(x_s, y_s, fraction)
            drv = str(row["driver"])
            color = DRIVER_COLORS.get(drv, "#AAAAAA")
            pos = int(row["position"])
            fig.add_trace(go.Scatter(
                x=[x_d], y=[y_d],
                mode="markers+text",
                marker=dict(size=13, color=color, symbol="circle",
                            line=dict(color="white", width=1.5)),
                text=[drv],
                textposition="top center",
                textfont=dict(color="white", size=9),
                name=f"P{pos} {drv}",
                showlegend=True,
                hovertext=f"P{pos} — {drv}",
                hoverinfo="text",
            ))

    fig.update_layout(
        paper_bgcolor="#0F0F0F",
        plot_bgcolor="#0F0F0F",
        xaxis=dict(visible=False, range=[x_min, x_max], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[y_min, y_max]),
        margin=dict(l=10, r=10, t=45, b=10),
        height=320,
        title=dict(text=f"<b>{path_data['name']}</b>", font=dict(color="white", size=13), x=0.5),
        legend=dict(
            orientation="h", yanchor="top", y=-0.05, x=0.0,
            font=dict(color="white", size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def _get_circuit_profile(race_slug: str) -> dict:
    for key in CIRCUIT_PROFILES:
        if key.split("_")[0] in race_slug:
            return CIRCUIT_PROFILES.get(key, CIRCUIT_PROFILES["bahrain_2025"])
    return CIRCUIT_PROFILES.get(race_slug, CIRCUIT_PROFILES["bahrain_2025"])


def _sector_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-driver sector time delta vs field median (ms, lower = faster)."""
    s_cols = ["sector1_time", "sector2_time", "sector3_time"]
    available = [c for c in s_cols if c in df.columns and df[c].notna().any()]
    if not available:
        return pd.DataFrame()

    rows = []
    for driver, grp in df.groupby("driver"):
        row = {"driver": driver}
        for col in available:
            field_median = df[col].median()
            driver_median = grp[col].median()
            if pd.notna(driver_median) and pd.notna(field_median) and field_median > 0:
                delta_ms = (driver_median - field_median) * 1000
                row[col.replace("_time", "")] = round(delta_ms, 1)
            else:
                row[col.replace("_time", "")] = 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("driver")


def _grid_finish_df(all_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compile grid position and final race position for each driver across all races."""
    rows = []
    for slug, df in all_dfs.items():
        if "grid_position" not in df.columns:
            continue
        for driver, grp in df.groupby("driver"):
            grid = grp["grid_position"].iloc[0] if pd.notna(grp["grid_position"].iloc[0]) else None
            final_pos = grp["position"].dropna()
            finish = int(final_pos.iloc[-1]) if len(final_pos) > 0 else None
            if grid and finish:
                rows.append({
                    "race": slug,
                    "driver": driver,
                    "grid": int(grid),
                    "finish": finish,
                    "gained": int(grid) - finish,
                })
    return pd.DataFrame(rows)


def render_track_intel(df: pd.DataFrame, race_slug: str, all_race_dfs: dict, driver: str, lap: int, mode: str):
    circuit = _get_circuit_profile(race_slug)

    # ── Circuit Profile Card ───────────────────────────────────────────────────
    st.subheader(f"Track Intel — {circuit.get('track_character', '').split('.')[0]}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Circuit Length", f"{circuit['length_km']} km")
    c2.metric("Race Laps", circuit["laps"])
    c3.metric("DRS Zones", circuit["drs_zones"])
    difficulty = circuit["overtaking_difficulty"]
    diff_label = "Very Easy" if difficulty < 30 else "Easy" if difficulty < 50 else "Moderate" if difficulty < 70 else "Hard" if difficulty < 90 else "Impossible"
    c4.metric("Overtaking", f"{difficulty}/100 — {diff_label}")

    st.markdown(f"**Key corners:** {', '.join(circuit['key_corners'])}")
    agg_sector_labels = [f"Sector {s}" for s in circuit["aggression_sectors"]]
    st.markdown(f"**Aggression pays in:** {', '.join(agg_sector_labels)}")
    st.markdown(circuit["track_character"])

    # Granite track brief
    sector_dom = _sector_dominance(df)
    sector_dom_dict = sector_dom.to_dict("index") if len(sector_dom) > 0 else {}
    granite_brief = track_intel_brief(circuit, sector_dom_dict, mode)
    mode_label = "Fan" if mode == "fan" else "Engineer"
    st.info(f"**{mode_label} Intel:** {granite_brief}")

    # ── Circuit Layout Diagram ─────────────────────────────────────────────────
    lap_df = df[df["lap"] == lap].copy() if "lap" in df.columns else pd.DataFrame()
    fig_circuit = _render_circuit_map(race_slug, lap_df=lap_df)
    if fig_circuit:
        st.subheader("Circuit Layout")
        st.plotly_chart(fig_circuit, use_container_width=True)

    st.divider()

    # ── Sector Dominance Heatmap ───────────────────────────────────────────────
    st.subheader("Sector Dominance")
    st.caption("Delta vs. field median (ms) — negative = faster than average")

    if len(sector_dom) > 0:
        cols_avail = [c for c in ["sector1", "sector2", "sector3"] if c in sector_dom.columns]
        col_labels = {"sector1": "S1", "sector2": "S2", "sector3": "S3"}
        heat_df = sector_dom[cols_avail].rename(columns=col_labels)

        drivers_ordered = heat_df.index.tolist()
        sector_labels = heat_df.columns.tolist()
        z_vals = heat_df.values.tolist()

        fig_heat = go.Figure(go.Heatmap(
            z=z_vals,
            x=sector_labels,
            y=drivers_ordered,
            colorscale=[[0, "#E10600"], [0.5, "#1C1C1C"], [1, "#27F4D2"]],
            zmid=0,
            text=[[f"{v:+.0f}ms" for v in row] for row in z_vals],
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="ms vs median"),
        ))
        fig_heat.update_layout(
            paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
            font=dict(color="white"),
            height=260,
            margin=dict(l=60, r=20, t=10, b=40),
            xaxis=dict(title="Sector", side="bottom"),
            yaxis=dict(title="Driver", autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Sector time data not available for this race.")

    st.divider()

    # ── Grid → Finish Position Analysis ───────────────────────────────────────
    st.subheader("Starting Grid vs. Race Finish")

    gf_df = _grid_finish_df(all_race_dfs)

    if len(gf_df) > 0:
        col_left, col_right = st.columns(2)

        # Selected race scatter
        with col_left:
            race_gf = gf_df[gf_df["race"] == race_slug]
            if len(race_gf) > 0:
                st.caption("This race — grid vs finish position")
                fig_scatter = go.Figure()
                for _, row in race_gf.iterrows():
                    drv = row["driver"]
                    color = DRIVER_COLORS.get(drv, "#888888")
                    fig_scatter.add_trace(go.Scatter(
                        x=[row["grid"]], y=[row["finish"]],
                        mode="markers+text",
                        name=drv,
                        marker=dict(size=14, color=color),
                        text=[drv], textposition="top center",
                        showlegend=False,
                    ))
                # Diagonal line (grid == finish)
                max_pos = max(race_gf["grid"].max(), race_gf["finish"].max()) + 1
                fig_scatter.add_trace(go.Scatter(
                    x=list(range(1, max_pos)), y=list(range(1, max_pos)),
                    mode="lines", line=dict(color="#555555", dash="dot"),
                    name="No change", showlegend=False,
                ))
                fig_scatter.update_layout(
                    paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
                    font=dict(color="white"),
                    xaxis=dict(title="Grid Position", gridcolor="#2A2A2A", autorange="reversed"),
                    yaxis=dict(title="Finish Position", gridcolor="#2A2A2A", autorange="reversed"),
                    height=280,
                    margin=dict(l=50, r=20, t=10, b=50),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Season average bar
        with col_right:
            st.caption("Season average — positions gained from grid")
            avg_gained = gf_df.groupby("driver")["gained"].mean().reset_index()
            avg_gained = avg_gained.sort_values("gained", ascending=True)
            colors = ["#E10600" if g < 0 else "#27F4D2" for g in avg_gained["gained"]]
            fig_bar = go.Figure(go.Bar(
                x=avg_gained["gained"],
                y=avg_gained["driver"],
                orientation="h",
                marker_color=colors,
                text=[f"{g:+.1f}" for g in avg_gained["gained"]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
                font=dict(color="white"),
                xaxis=dict(title="Avg positions gained", gridcolor="#2A2A2A", zeroline=True,
                           zerolinecolor="#888888"),
                yaxis=dict(gridcolor="#2A2A2A"),
                height=280,
                margin=dict(l=60, r=60, t=10, b=50),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Grid position data not available.")

    st.divider()

    # ── Aggression Zones ───────────────────────────────────────────────────────
    st.subheader("Where in the Race Do Position Battles Happen?")

    agg_stats = get_aggression_zone_stats(df)
    if len(agg_stats) > 0:
        fig_agg = go.Figure()
        zone_colors = {"Early": "#E10600", "Mid": "#FF8000", "Late": "#27F4D2"}
        for zone_col, color in zone_colors.items():
            if zone_col in agg_stats.columns:
                fig_agg.add_trace(go.Bar(
                    name=f"{zone_col} laps",
                    x=agg_stats["driver"],
                    y=agg_stats[zone_col],
                    marker_color=color,
                ))
        fig_agg.update_layout(
            barmode="group",
            paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
            font=dict(color="white"),
            xaxis=dict(title="Driver", gridcolor="#2A2A2A"),
            yaxis=dict(title="Position changes", gridcolor="#2A2A2A"),
            height=260,
            margin=dict(l=50, r=20, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
        )
        st.plotly_chart(fig_agg, use_container_width=True)

    st.divider()

    # ── Battle Prediction ──────────────────────────────────────────────────────
    st.subheader(f"Battle Forecast — Next 15 Laps from Lap {lap}")

    if len(df) == 0:
        st.info("No race data for battle forecast.")
        return

    n_laps = 15
    proj_df, aggression_windows = forecast_positions(df, lap, n_laps)

    if len(proj_df) == 0:
        st.info("Not enough data for battle forecast at this lap.")
        return

    # Position trajectory chart
    fig_traj = go.Figure()
    drivers_in_proj = proj_df["driver"].unique()
    for drv in drivers_in_proj:
        drv_proj = proj_df[proj_df["driver"] == drv]
        color = DRIVER_COLORS.get(drv, "#888888")
        is_selected = drv == driver
        fig_traj.add_trace(go.Scatter(
            x=drv_proj["lap"],
            y=drv_proj["projected_position"],
            name=drv,
            line=dict(
                color=color,
                width=3 if is_selected else 1.5,
                dash="solid" if is_selected else "dot",
            ),
            mode="lines",
        ))

    fig_traj.add_vline(x=lap, line=dict(color="white", dash="dot", width=1),
                       annotation_text="Now", annotation_position="top left",
                       annotation_font=dict(color="white", size=10))

    fig_traj.update_layout(
        paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
        font=dict(color="white"),
        xaxis=dict(title="Lap", gridcolor="#2A2A2A"),
        yaxis=dict(title="Projected Position", autorange="reversed",
                   tickvals=list(range(1, len(drivers_in_proj) + 1)),
                   gridcolor="#2A2A2A"),
        height=280,
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # Aggression windows table
    if aggression_windows:
        drs_windows = [w for w in aggression_windows if w["is_drs_range"]]
        close_windows = [w for w in aggression_windows if not w["is_drs_range"]]

        if drs_windows:
            st.markdown("**DRS range battles (gap < 1.0s):**")
            for w in drs_windows[:5]:
                st.markdown(
                    f"- Lap {w['lap']}: **{w['driver_behind']}** vs {w['driver_ahead']} "
                    f"— projected gap {w['projected_gap']:.2f}s"
                )
        if close_windows:
            st.markdown("**Close battles (gap 1-2s):**")
            for w in close_windows[:5]:
                st.markdown(
                    f"- Lap {w['lap']}: {w['driver_behind']} approaching {w['driver_ahead']} "
                    f"({w['projected_gap']:.2f}s)"
                )
    else:
        st.write("No close battles forecast in the next 15 laps.")

    st.divider()

    # ── Pilot Error & Strategy Oversight ──────────────────────────────────────
    st.subheader("Pilot Error & Strategy Oversight")
    st.caption("Detected from telemetry: lock-ups, pace collapses, late pits, SC restart losses, radio stress")

    # Derive flag periods without importing load_flags (avoids sys.modules cache issues)
    try:
        import os as _os, json as _json
        _cache = _os.path.join(_os.path.dirname(__file__), "..", "data", "cache", f"{race_slug}_flags.json")
        flag_periods = _json.load(open(_cache)) if _os.path.exists(_cache) else []
    except Exception:
        flag_periods = []
    if not flag_periods and "safety_car_active" in df.columns:
        sc_laps = sorted(df[df["safety_car_active"]]["lap"].unique())
        flag_periods, start, prev = [], sc_laps[0] if sc_laps else None, sc_laps[0] if sc_laps else None
        for lap in (sc_laps[1:] if sc_laps else []):
            if lap > prev + 1:
                flag_periods.append({"flag": "SAFETY_CAR", "lap_start": int(start), "lap_end": int(prev)})
                start = lap
            prev = lap
        if sc_laps:
            flag_periods.append({"flag": "SAFETY_CAR", "lap_start": int(start), "lap_end": int(prev)})
    error_df = detect_pilot_errors(df, flag_periods)
    counts = error_counts(error_df)

    mode_label = "Fan" if mode == "fan" else "Engineer"

    if len(counts) == 0:
        st.info("No pilot errors detected for this race (may require radio sentiment data for full coverage).")
    else:
        # Summary row — all drivers
        st.markdown("**Error Summary — All Drivers**")
        summary_cols = st.columns(min(len(counts), 4))
        for i, (drv, type_counts) in enumerate(sorted(counts.items(), key=lambda x: sum(x[1].values()), reverse=True)):
            total = sum(type_counts.values())
            col = summary_cols[i % len(summary_cols)]
            col.metric(drv, f"{total} errors", " · ".join(f"{v} {k.replace('_',' ').lower()}" for k, v in type_counts.items()))

        st.divider()

        # Selected driver detail
        st.markdown(f"**{driver} — Incident Log**")
        drv_errors = error_df[error_df["driver"] == driver] if len(error_df) > 0 else pd.DataFrame()

        # Granite summary
        drv_error_list = drv_errors.to_dict("records") if len(drv_errors) > 0 else []
        granite_err = error_summary(drv_error_list, driver, mode)
        st.info(f"**{mode_label} Analysis:** {granite_err}")

        if len(drv_errors) > 0:
            ERROR_ICONS = {
                "LOCKUP": "⚡", "PACE_COLLAPSE": "📉", "LATE_PIT": "🔧",
                "SC_LOSS": "🚨", "RADIO_STRESS": "📻", "TRACK_LIMITS": "⚠️",
            }
            for _, row in drv_errors.iterrows():
                icon = ERROR_ICONS.get(row["error_type"], "•")
                severity_color = "🔴" if row["severity"] == "major" else "🟡"
                st.markdown(
                    f"{severity_color} **Lap {int(row['lap'])}** {icon} `{row['error_type']}` — "
                    f"{row['description']} *(signal: {row['signal_value']:.2f})*"
                )
        else:
            st.success(f"{driver} had a clean race — no errors detected.")

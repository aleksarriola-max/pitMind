"""Track Intel — circuit profile, sector dominance, grid analysis, battle prediction."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from models.race_forecast import forecast_positions, get_aggression_zone_stats

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
    from agent.granite import track_intel_brief
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

    from models.error_detection import detect_pilot_errors, error_counts, total_errors
    from data.ingest import load_flags
    from agent.granite import error_summary

    flag_periods = load_flags(race_slug, df)
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

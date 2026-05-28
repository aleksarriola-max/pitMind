"""Driver Statistics Tab — season leaderboard, H2H radar, season arc."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.ingest import DRIVERS

try:
    from agent.granite import h2h_narrative
except ImportError:
    def h2h_narrative(a, b, sa, sb, mode="fan"):
        return f"Head-to-head comparison: {a} vs {b}. Granite narrative unavailable."

try:
    from models.driver_stats import compute_season_stats, season_arc, rank_drivers
except ImportError:
    def compute_season_stats(dfs): import pandas as pd; return pd.DataFrame()
    def season_arc(dfs, driver): import pandas as pd; return pd.DataFrame()
    def rank_drivers(df): return df

DRIVER_COLORS = {
    "VER": "#3671C6", "HAM": "#E8002D", "LEC": "#E8002D",
    "PIA": "#FF8000", "ALO": "#229971", "OCO": "#B6BABD",
    "NOR": "#FF8000", "RUS": "#27F4D2",
}

LEADERBOARD_METRICS = {
    "avg_finish":          ("Avg Finish", True),   # (label, lower_is_better)
    "avg_grid":            ("Avg Grid",   True),
    "avg_positions_gained":("Pos Gained", False),
    "avg_pace_delta":      ("Pace Δ (s)", True),
    "tyre_preservation":   ("Tyre Save",  False),
    "aggression_level":    ("Aggression", False),
    "overtake_tendency":   ("Overtakes",  False),
    "pressure_consistency":("Consistency",False),
    "pit_compliance":      ("Pit Comply", False),
}

RADAR_METRICS = [
    "tyre_preservation", "aggression_level", "overtake_tendency",
    "pressure_consistency", "restart_aggression", "dirty_air_tolerance",
    "pit_compliance", "late_braking_tendency",
]

RADAR_LABELS = [
    "Tyre Save", "Aggression", "Overtakes",
    "Consistency", "SC Restart", "Dirty Air",
    "Pit Comply", "Late Brake",
]


def render_driver_stats(all_race_dfs: dict, selected_driver: str, mode: str):
    st.subheader("Driver Statistics — 2025 Season")

    stats_df = compute_season_stats(all_race_dfs)
    if len(stats_df) == 0:
        st.info("Not enough race data to compute season statistics.")
        return

    stats_df = rank_drivers(stats_df)
    mode_label = "Fan" if mode == "fan" else "Engineer"

    # ── Section 1: Leaderboard ─────────────────────────────────────────────────
    st.markdown("### Driver Leaderboard")
    st.caption("Click any column header to sort · Green = top 3 · Red = bottom 3")

    FAN_LEADERBOARD_COLS = ["avg_finish", "avg_positions_gained", "aggression_level", "tyre_preservation"]
    metrics_to_show = LEADERBOARD_METRICS if mode == "engineer" else {
        k: v for k, v in LEADERBOARD_METRICS.items() if k in FAN_LEADERBOARD_COLS
    }
    display_cols = {}
    for col, (label, lower_better) in metrics_to_show.items():
        if col in stats_df.columns:
            display_cols[col] = label

    if display_cols:
        leaderboard = stats_df[list(display_cols.keys())].copy()
        # Convert all metric columns to float, replacing None/NaN with empty string for display
        for col in leaderboard.columns:
            leaderboard[col] = pd.to_numeric(leaderboard[col], errors="coerce")
        leaderboard.columns = list(display_cols.values())
        leaderboard = leaderboard.reset_index()
        leaderboard.insert(0, "#", range(1, len(leaderboard) + 1))

        # Format numerics to 1 decimal; leave NaN as "—"
        numeric_cols = [c for c in leaderboard.columns if c not in ("#", "driver")]
        for col in numeric_cols:
            leaderboard[col] = leaderboard[col].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—"
            )

        st.dataframe(leaderboard, use_container_width=True, height=320, hide_index=True)

    st.divider()

    # ── Section 2: H2H Comparison ──────────────────────────────────────────────
    st.markdown("### Head-to-Head Comparison")

    available = [d for d in DRIVERS if d in stats_df.index]
    col_a, col_b = st.columns(2)
    driver_a = col_a.selectbox("Driver A", options=available,
                               index=available.index(selected_driver) if selected_driver in available else 0,
                               key="h2h_a")
    driver_b = col_b.selectbox("Driver B", options=available,
                               index=min(1, len(available) - 1) if len(available) > 1 else 0, key="h2h_b")

    if driver_a != driver_b:
        radar_vals = []
        for drv, col_key in [(driver_a, "a"), (driver_b, "b")]:
            vals = []
            for metric in RADAR_METRICS:
                if metric in stats_df.columns and drv in stats_df.index:
                    v = stats_df.loc[drv, metric]
                    vals.append(float(v) if pd.notna(v) else 50.0)
                else:
                    vals.append(50.0)
            radar_vals.append(vals)

        fig_radar = go.Figure()
        for drv, vals, color in [(driver_a, radar_vals[0], DRIVER_COLORS.get(driver_a, "#AAAAAA")),
                                  (driver_b, radar_vals[1], DRIVER_COLORS.get(driver_b, "#FFFFFF"))]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=RADAR_LABELS + [RADAR_LABELS[0]],
                fill="toself",
                name=drv,
                line=dict(color=color),
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)" if color.startswith("#") else color.replace(")", ",0.15)").replace("rgb", "rgba"),
                opacity=0.85,
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1C1C1C",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333", tickfont=dict(color="#888")),
                angularaxis=dict(gridcolor="#333", tickfont=dict(color="white")),
            ),
            paper_bgcolor="#0F0F0F",
            font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.05),
            height=380,
            margin=dict(l=50, r=50, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Granite H2H narrative
        stats_a = {k: stats_df.loc[driver_a, k] for k in LEADERBOARD_METRICS if k in stats_df.columns and driver_a in stats_df.index}
        stats_b = {k: stats_df.loc[driver_b, k] for k in LEADERBOARD_METRICS if k in stats_df.columns and driver_b in stats_df.index}
        narrative = h2h_narrative(driver_a, driver_b,
                                  {k: (float(v) if pd.notna(v) else None) for k, v in stats_a.items()},
                                  {k: (float(v) if pd.notna(v) else None) for k, v in stats_b.items()},
                                  mode)
        st.info(f"**{mode_label} H2H:** {narrative}")

        # Win/loss table per metric (engineer only)
        if mode == "engineer":
            st.markdown("**Metric-by-metric breakdown:**")
            rows = []
            for col, (label, lower_better) in LEADERBOARD_METRICS.items():
                if col not in stats_df.columns:
                    continue
                va = stats_df.loc[driver_a, col] if driver_a in stats_df.index else None
                vb = stats_df.loc[driver_b, col] if driver_b in stats_df.index else None
                if pd.isna(va) or pd.isna(vb):
                    continue
                if lower_better:
                    winner = driver_a if va < vb else driver_b
                else:
                    winner = driver_a if va > vb else driver_b
                rows.append({"Metric": label, driver_a: f"{float(va):.1f}", driver_b: f"{float(vb):.1f}", "Edge": winner})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 3: Season Arc ──────────────────────────────────────────────────
    st.markdown(f"### {selected_driver} — Season Arc")
    st.caption("Performance trajectory across the 5 races")

    arc_df = season_arc(all_race_dfs, selected_driver)
    if len(arc_df) == 0:
        st.info(f"No season arc data for {selected_driver}.")
        return

    arc_metrics = (
        [m for m in ["finish_position", "avg_pace_delta", "tyre_preservation",
                     "aggression_level", "pressure_consistency"] if m in arc_df.columns]
        if mode == "engineer"
        else [m for m in ["finish_position"] if m in arc_df.columns]
    )
    metric_labels = {
        "finish_position": "Finish Position",
        "avg_pace_delta": "Avg Pace Delta (s)",
        "tyre_preservation": "Tyre Preservation",
        "aggression_level": "Aggression Level",
        "pressure_consistency": "Pressure Consistency",
    }

    arc_col1, arc_col2 = st.columns(2)
    driver_color = DRIVER_COLORS.get(selected_driver, "#AAAAAA")

    for i, metric in enumerate(arc_metrics):
        col = arc_col1 if i % 2 == 0 else arc_col2
        vals = pd.to_numeric(arc_df[metric], errors="coerce")
        if vals.isna().all():
            continue
        fig_arc = go.Figure()
        fig_arc.add_trace(go.Scatter(
            x=arc_df["race"],
            y=vals,
            mode="lines+markers",
            line=dict(color=driver_color, width=2),
            marker=dict(size=8, color=driver_color),
            name=metric_labels.get(metric, metric),
        ))
        invert = metric in ("finish_position", "avg_pace_delta")
        fig_arc.update_layout(
            paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
            font=dict(color="white"),
            xaxis=dict(title="Race", gridcolor="#2A2A2A", tickangle=-20),
            yaxis=dict(title=metric_labels.get(metric, metric), gridcolor="#2A2A2A",
                       autorange="reversed" if invert else True),
            height=200,
            margin=dict(l=50, r=20, t=20, b=60),
            showlegend=False,
        )
        with col:
            st.caption(metric_labels.get(metric, metric))
            st.plotly_chart(fig_arc, use_container_width=True)

    # ── Section 4: Intra-Team Comparison ──────────────────────────────────────
    if mode == "engineer":
        st.divider()
        st.markdown("### Intra-Team Comparison")
        st.caption("Head-to-head between teammates across the 2025 season")

        # Build team → [drivers in stats_df] mapping
        DRIVER_TO_TEAM = {
            "VER": "Red Bull",    "OCO": "Haas",
            "HAM": "Ferrari",     "LEC": "Ferrari",
            "PIA": "McLaren",     "RUS": "Mercedes",
            "NOR": "McLaren",     "ALO": "Aston Martin",
        }
        from collections import defaultdict
        team_map = defaultdict(list)
        for drv in stats_df.index:
            team = DRIVER_TO_TEAM.get(drv)
            if team:
                team_map[team].append(drv)

        teammate_pairs = [(team, drivers) for team, drivers in team_map.items() if len(drivers) >= 2]

        if not teammate_pairs:
            st.info("No intra-team data available — need at least 2 teammates in the dataset.")
        else:
            for team, drivers in teammate_pairs:
                st.markdown(f"**{team} — {' vs '.join(drivers)}**")
                drv_a, drv_b = drivers[0], drivers[1]

                rows = []
                for col, (label, lower_better) in LEADERBOARD_METRICS.items():
                    if col not in stats_df.columns:
                        continue
                    va = stats_df.loc[drv_a, col] if drv_a in stats_df.index else None
                    vb = stats_df.loc[drv_b, col] if drv_b in stats_df.index else None
                    if pd.isna(va) or pd.isna(vb):
                        continue
                    if lower_better:
                        winner = drv_a if va < vb else drv_b
                        edge = abs(float(va) - float(vb))
                    else:
                        winner = drv_a if va > vb else drv_b
                        edge = abs(float(va) - float(vb))
                    rows.append({
                        "Metric": label,
                        drv_a: f"{float(va):.1f}",
                        drv_b: f"{float(vb):.1f}",
                        "Edge": f"{winner} +{edge:.1f}",
                    })

                if rows:
                    team_df = pd.DataFrame(rows)
                    st.dataframe(team_df, use_container_width=True, hide_index=True)

                    # Win count summary
                    wins_a = sum(1 for r in rows if r["Edge"].startswith(drv_a))
                    wins_b = sum(1 for r in rows if r["Edge"].startswith(drv_b))
                    overall = drv_a if wins_a >= wins_b else drv_b
                    st.caption(f"**{drv_a}** leads {wins_a} metrics · **{drv_b}** leads {wins_b} metrics · Overall edge: **{overall}**")

                st.markdown("")

    # ── Data Export ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Export Race Data")
    st.caption("Download the underlying season statistics as CSV for your own analysis")

    if len(stats_df) > 0:
        export_df = stats_df.reset_index()
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Season Stats (CSV)",
            data=csv_bytes,
            file_name="pitmind_season_stats_2025.csv",
            mime="text/csv",
            help="Exports all driver metrics across the 2025 season",
        )

    # Export individual race data
    if all_race_dfs:
        race_options = list(all_race_dfs.keys())
        export_race = st.selectbox("Export individual race data", race_options, key="export_race_select")
        if export_race and export_race in all_race_dfs:
            race_df_export = all_race_dfs[export_race]
            if len(race_df_export) > 0:
                race_csv = race_df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇ Download {export_race} race data (CSV)",
                    data=race_csv,
                    file_name=f"pitmind_{export_race.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    help=f"Full per-lap telemetry data for {export_race}",
                )

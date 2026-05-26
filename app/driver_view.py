"""Driver Soul — radar chart, live behavioral state, predictions, SHAP."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from models.driver_soul import TRAIT_COLS, PRED_COLS, get_driver_fingerprint, get_lap_state
from agent.granite import explain_driver

TRAIT_LABELS = {
    "overtake_tendency": "Overtake Tendency",
    "position_vulnerability": "Position Vulnerability",
    "aggression_level": "Aggression Level",
    "tyre_preservation": "Tyre Preservation",
    "pit_compliance": "Pit Compliance",
    "restart_aggression": "Restart Aggression",
    "pressure_consistency": "Pressure Consistency",
    "late_braking_tendency": "Late Braking",
    "radio_stress_frequency": "Radio Stress Freq.",
    "undercut_susceptibility": "Undercut Susceptibility",
    "dirty_air_tolerance": "Dirty Air Tolerance",
}

PRED_LABELS = {
    "position_gain_prob": "Position Gain Prob.",
    "position_loss_prob": "Position Loss Prob.",
    "pit_prob": "Pit Probability",
    "incident_risk": "Incident Risk",
}

PRED_COLORS = {
    "position_gain_prob": "#22C55E",
    "position_loss_prob": "#EF4444",
    "pit_prob": "#F59E0B",
    "incident_risk": "#8B5CF6",
}


def _radar_chart(fingerprint: dict, driver: str):
    labels = [TRAIT_LABELS.get(k, k) for k in TRAIT_COLS]
    values = [fingerprint.get(k, 50) for k in TRAIT_COLS]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(225, 6, 0, 0.15)",
        line=dict(color="#E10600", width=2),
        name=driver,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1A1A1A",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333", tickfont=dict(size=8, color="#888")),
            angularaxis=dict(gridcolor="#333", tickfont=dict(size=9, color="white")),
        ),
        paper_bgcolor="#0F0F0F",
        font=dict(color="white"),
        showlegend=False,
        height=380,
        margin=dict(l=60, r=60, t=30, b=30),
    )
    return fig


def _prediction_bars(lap_state: dict):
    fig = go.Figure()
    for col in PRED_COLS:
        val = lap_state.get(col, 0.5)
        label = PRED_LABELS.get(col, col)
        color = PRED_COLORS.get(col, "#AAAAAA")
        fig.add_trace(go.Bar(
            x=[val],
            y=[label],
            orientation="h",
            marker=dict(color=color, opacity=0.85),
            text=[f"{val:.0%}"],
            textposition="outside",
            name=label,
        ))
    fig.update_layout(
        paper_bgcolor="#0F0F0F",
        plot_bgcolor="#0F0F0F",
        font=dict(color="white"),
        xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#2A2A2A"),
        yaxis=dict(gridcolor="#2A2A2A"),
        showlegend=False,
        height=200,
        margin=dict(l=160, r=60, t=10, b=30),
        barmode="overlay",
    )
    return fig


def _shap_bars(shap_vals: dict):
    if not shap_vals:
        return None
    features = list(shap_vals.keys())
    values = list(shap_vals.values())
    colors = ["#22C55E" if v > 0 else "#EF4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=[f.replace("_", " ").title() for f in features],
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#0F0F0F",
        plot_bgcolor="#0F0F0F",
        font=dict(color="white"),
        xaxis=dict(title="SHAP value (impact on position gain prediction)", gridcolor="#2A2A2A"),
        yaxis=dict(gridcolor="#2A2A2A"),
        showlegend=False,
        height=180,
        margin=dict(l=160, r=60, t=10, b=40),
    )
    return fig


@st.cache_data(show_spinner=False)
def _get_shap_for_lap(driver: str, lap: int, race_slug: str) -> dict:
    """Get SHAP values for a specific driver+lap from saved model."""
    try:
        import joblib, numpy as np
        from models.driver_soul import MODELS_DIR, load_models
        from data.ingest import load_race
        feature_cols = TRAIT_COLS + ["tyre_age", "gap_ahead", "gap_behind", "pit_window_pressure", "lap_time"]
        df = load_race(race_slug)
        row = df[(df["driver"] == driver) & (df["lap"] == lap)]
        if len(row) == 0:
            return {}
        models = load_models()
        explainer = models.get("shap_explainer")
        feature_names = models.get("feature_names", feature_cols)
        if explainer is None:
            return {}
        available = [c for c in feature_names if c in row.columns]
        X = row[available].fillna(0).values
        from models.driver_soul import get_shap_values
        return get_shap_values(explainer, X[0], available)
    except Exception as e:
        return {}


@st.cache_data(show_spinner=False)
def _get_narrative(driver: str, fingerprint_json: str, shap_json: str, mode: str) -> str:
    import json
    fingerprint = json.loads(fingerprint_json)
    shap_vals = json.loads(shap_json)
    return explain_driver(fingerprint, shap_vals, driver, mode)


def render_driver_soul(df: pd.DataFrame, driver: str, lap: int, mode: str):
    st.subheader(f"Driver Soul — {driver}")

    if len(df) == 0 or driver not in df["driver"].unique():
        st.info(f"No data for driver {driver}.")
        return

    race_slug = df.attrs.get("race_slug", "bahrain_2023")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Behavioral Fingerprint** (season average)")
        fingerprint = get_driver_fingerprint(df, driver)
        fig = _radar_chart(fingerprint, driver)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown(f"**Live State at Lap {lap}**")
        lap_state = get_lap_state(df, driver, lap)

        if not lap_state:
            st.info(f"No data for {driver} at lap {lap}.")
        else:
            # Trait scores as small metric cards (2 columns)
            st.markdown("*Behavioral Traits (0–100)*")
            cols = st.columns(2)
            for i, trait in enumerate(TRAIT_COLS):
                val = lap_state.get(trait, 0)
                label = TRAIT_LABELS.get(trait, trait)
                bar = "█" * int(val / 10) + "░" * (10 - int(val / 10))
                cols[i % 2].markdown(f"**{label}**  \n`{bar}` {val:.0f}")

    # Predictions
    st.divider()
    st.markdown(f"**Race Outcome Predictions at Lap {lap}**")
    if lap_state:
        fig_pred = _prediction_bars(lap_state)
        st.plotly_chart(fig_pred, use_container_width=True)
        if mode == "engineer":
            pred_rows = [{"Outcome": PRED_LABELS.get(c, c), "Probability": f"{lap_state.get(c, 0):.1%}"} for c in PRED_COLS]
            st.dataframe(pd.DataFrame(pred_rows), hide_index=True, use_container_width=True)

    # SHAP (engineer only)
    st.divider()
    if mode == "engineer":
        st.markdown("**SHAP — What's driving the position gain prediction?**")
        shap_vals = _get_shap_for_lap(driver, lap, df["race"].iloc[0].lower().replace(" ", "_"))
        if shap_vals:
            fig_shap = _shap_bars(shap_vals)
            if fig_shap:
                st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.caption("SHAP values unavailable for this lap.")

        # Model feature importance (global, from trained predictor)
        st.divider()
        st.markdown("**Model Feature Importance — Position Gain Predictor**")
        try:
            from models.driver_soul import load_models
            _models = load_models()
            _predictor = _models.get("predictor")
            _feat_names = _models.get("feature_names", [])
            if _predictor is not None and hasattr(_predictor, "estimators_") and _feat_names:
                _importances = _predictor.estimators_[0].feature_importances_
                _imp_pairs = sorted(zip(_feat_names, _importances), key=lambda x: x[1], reverse=True)[:10]
                _labels = [p[0].replace("_", " ").title() for p in _imp_pairs]
                _vals = [p[1] for p in _imp_pairs]
                _fig_imp = go.Figure(go.Bar(
                    x=_vals,
                    y=_labels,
                    orientation="h",
                    marker=dict(color="#3671C6", opacity=0.85),
                    text=[f"{v:.3f}" for v in _vals],
                    textposition="outside",
                ))
                _fig_imp.update_layout(
                    paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
                    font=dict(color="white"),
                    xaxis=dict(title="Feature Importance", gridcolor="#2A2A2A"),
                    yaxis=dict(gridcolor="#2A2A2A"),
                    showlegend=False,
                    height=280,
                    margin=dict(l=160, r=60, t=10, b=40),
                )
                st.plotly_chart(_fig_imp, use_container_width=True)
                st.caption("Global XGBoost feature importances for the position gain prediction model")
            else:
                st.caption("Feature importance unavailable.")
        except Exception:
            st.caption("Feature importance unavailable.")
    else:
        shap_vals = {}

    # Engineer-only telemetry trend charts
    if mode == "engineer":
        drv_all = df[df["driver"] == driver].sort_values("lap")

        st.divider()
        st.markdown("**Radio Sentiment Trend**")
        sent = drv_all[["lap", "radio_sentiment"]].dropna()
        if len(sent) > 0:
            fig_sent = go.Figure(go.Scatter(
                x=sent["lap"], y=sent["radio_sentiment"],
                mode="lines", line=dict(color="#F59E0B", width=2),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
                hovertemplate="Lap %{x}<br>Sentiment: %{y:.2f}<extra></extra>",
            ))
            fig_sent.add_hline(y=0, line=dict(color="#555555", dash="dot", width=1))
            fig_sent.update_layout(
                paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
                font=dict(color="white"),
                xaxis=dict(title="Lap", gridcolor="#2A2A2A"),
                yaxis=dict(title="Sentiment (−1 stress → +1 calm)", range=[-1.1, 1.1], gridcolor="#2A2A2A"),
                height=200, margin=dict(l=60, r=20, t=10, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.caption("Radio sentiment data unavailable.")

        st.divider()
        st.markdown("**Incident Risk Trend**")
        risk = drv_all[["lap", "incident_risk"]].dropna()
        if len(risk) > 0:
            fig_risk = go.Figure(go.Scatter(
                x=risk["lap"], y=risk["incident_risk"],
                mode="lines", line=dict(color="#EF4444", width=2),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
                hovertemplate="Lap %{x}<br>Risk: %{y:.2f}<extra></extra>",
            ))
            fig_risk.add_hline(y=0.5, line=dict(color="#EF4444", dash="dot", width=1),
                               annotation_text="High Risk", annotation_position="right",
                               annotation_font=dict(color="#EF4444", size=9))
            fig_risk.update_layout(
                paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
                font=dict(color="white"),
                xaxis=dict(title="Lap", gridcolor="#2A2A2A"),
                yaxis=dict(title="Incident Risk (0–1)", range=[0, 1], gridcolor="#2A2A2A"),
                height=200, margin=dict(l=60, r=20, t=10, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.caption("Incident risk data unavailable.")

        st.divider()
        st.markdown("**Prediction Probability Trends**")
        trend_cols = [c for c in PRED_COLS if c in drv_all.columns]
        if trend_cols:
            fig_trend = go.Figure()
            for col in trend_cols:
                trend_data = drv_all[["lap", col]].dropna()
                if len(trend_data) == 0:
                    continue
                fig_trend.add_trace(go.Scatter(
                    x=trend_data["lap"], y=trend_data[col],
                    mode="lines", name=PRED_LABELS.get(col, col),
                    line=dict(color=PRED_COLORS.get(col, "#AAAAAA"), width=2),
                    hovertemplate=f"{PRED_LABELS.get(col, col)}: %{{y:.0%}}<extra></extra>",
                ))
            fig_trend.update_layout(
                paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F",
                font=dict(color="white"),
                xaxis=dict(title="Lap", gridcolor="#2A2A2A"),
                yaxis=dict(title="Probability", range=[0, 1], tickformat=".0%", gridcolor="#2A2A2A"),
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
                height=220, margin=dict(l=60, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.caption("Prediction trend data unavailable.")

        # What would another driver do?
        st.divider()
        st.markdown("**What would another driver do in this situation?**")
        st.caption("Applies a different driver's behavioral model to the current lap context")

        other_drivers = [d for d in df["driver"].unique() if d != driver]
        if other_drivers:
            compare_driver = st.selectbox(
                "Compare with", other_drivers, key="soul_compare_driver"
            )
            compare_state = get_lap_state(df, compare_driver, lap)

            if compare_state and lap_state:
                # Side-by-side prediction comparison
                st.markdown(f"*{driver} vs {compare_driver} — predictions at lap {lap}*")
                cmp_cols = st.columns(len(PRED_COLS))
                for col, pred in zip(cmp_cols, PRED_COLS):
                    v_a = lap_state.get(pred, 0)
                    v_b = compare_state.get(pred, 0)
                    label = PRED_LABELS.get(pred, pred)
                    delta = v_b - v_a
                    col.metric(
                        label,
                        f"{v_a:.0%}",
                        f"{compare_driver}: {v_b:.0%} ({delta:+.0%})",
                        delta_color="inverse" if pred in ("position_loss_prob", "incident_risk") else "normal",
                    )

                # Pit decision comparison — the key insight
                pit_a = lap_state.get("pit_prob", 0.5)
                pit_b = compare_state.get("pit_prob", 0.5)
                st.markdown("---")
                insight_col1, insight_col2 = st.columns(2)
                insight_col1.metric(
                    f"{driver} pit probability",
                    f"{pit_a:.0%}",
                    "Would pit" if pit_a > 0.5 else "Would stay",
                    delta_color="off",
                )
                insight_col2.metric(
                    f"{compare_driver} pit probability",
                    f"{pit_b:.0%}",
                    "Would pit" if pit_b > 0.5 else "Would stay",
                    delta_color="off",
                )
                if abs(pit_a - pit_b) > 0.15:
                    diff_pct = abs(pit_a - pit_b)
                    more_likely = driver if pit_a > pit_b else compare_driver
                    less_likely = compare_driver if pit_a > pit_b else driver
                    st.info(
                        f"**{more_likely}** is **{diff_pct:.0%} more likely to pit** at this moment "
                        f"than **{less_likely}** — their behavioral profiles diverge significantly here."
                    )
            else:
                st.caption(f"No data for {compare_driver} at lap {lap}.")

    # Granite narrative
    st.divider()
    import json
    try:
        narrative = _get_narrative(driver, json.dumps(fingerprint), json.dumps(shap_vals), mode)
        mode_label = "Fan" if mode == "fan" else "Engineer"
        st.markdown(f"**{mode_label} Mode — Granite Analysis**")
        st.info(narrative)
    except Exception:
        pass

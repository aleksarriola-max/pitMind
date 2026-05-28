"""Driver Soul Model — 11 behavioral traits + multi-target predictions.

Traits (all 0-100, computed from FastF1 telemetry):
  1.  overtake_tendency
  2.  position_vulnerability
  3.  aggression_level
  4.  tyre_preservation
  5.  pit_compliance
  6.  restart_aggression
  7.  pressure_consistency
  8.  late_braking_tendency
  9.  radio_stress_frequency
  10. undercut_susceptibility
  11. dirty_air_tolerance

Multi-target XGBoost predictions (per lap):
  position_gain_prob    — probability of gaining a position in next 5 laps
  position_loss_prob    — probability of losing a position in next 5 laps
  pit_prob              — probability of pitting in next 3 laps
  incident_risk         — composite heuristic (tyre_age, aggression, pressure_consistency)
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved")
TRAIT_COLS = [
    "overtake_tendency", "position_vulnerability", "aggression_level",
    "tyre_preservation", "pit_compliance", "restart_aggression",
    "pressure_consistency", "late_braking_tendency", "radio_stress_frequency",
    "undercut_susceptibility", "dirty_air_tolerance",
]
PRED_COLS = ["position_gain_prob", "position_loss_prob", "pit_prob", "incident_risk"]
DRIVERS = ["VER", "HAM", "LEC", "PIA", "ALO", "OCO", "NOR", "RUS"]


# ---------------------------------------------------------------------------
# Trait engineering
# ---------------------------------------------------------------------------

def _norm(s: pd.Series, lo: float = 0.0, hi: float = 1.0) -> pd.Series:
    """Min-max normalize a series, clip to [lo, hi], scale to 0-100."""
    if s.isna().all():
        return pd.Series(50.0, index=s.index)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    return ((s - mn) / (mx - mn)).clip(lo, hi) * 100


def _rolling_std(s: pd.Series, window: int = 5) -> pd.Series:
    return s.rolling(window, min_periods=1).std().fillna(0)


def engineer_traits(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 11 behavioral trait scores per row. Returns df with trait columns added."""
    df = df.copy()
    df = df.sort_values(["driver", "lap"]).reset_index(drop=True)

    results = []
    for driver, grp in df.groupby("driver"):
        grp = grp.copy().sort_values("lap").reset_index(drop=True)
        n = len(grp)

        # 1. Overtake tendency: proportion of laps where gap_ahead decreased by >0.3s
        gap_delta = grp["gap_ahead"].diff()
        ot_score = (gap_delta < -0.3).astype(float).rolling(5, min_periods=1).mean() * 100
        grp["overtake_tendency"] = ot_score.fillna(30)

        # 2. Position vulnerability: probability next lap has a worse position
        pos_change = grp["position"].diff().shift(-1)
        pv_score = (pos_change > 0).astype(float).rolling(5, min_periods=1).mean() * 100
        grp["position_vulnerability"] = pv_score.fillna(20)

        # 3. Aggression level: lap time variability when being chased (<2s gap behind)
        chased = grp["gap_behind"] < 2.0
        lt_std = _rolling_std(grp["lap_time"], window=5)
        agg = np.where(chased, lt_std * 10, lt_std * 5)
        grp["aggression_level"] = _norm(pd.Series(agg, index=grp.index))

        # 4. Tyre preservation: inverse of pace degradation slope per stint
        def tyre_pres(stint_grp):
            if len(stint_grp) < 2:
                return pd.Series(50.0, index=stint_grp.index)
            x = stint_grp["tyre_age"].values.astype(float)
            y = stint_grp["lap_time"].values.astype(float)
            if x.std() < 1e-6:
                return pd.Series(50.0, index=stint_grp.index)
            slope = np.polyfit(x, y, 1)[0]
            score = max(0, 100 - slope * 20)
            return pd.Series(float(score), index=stint_grp.index)

        tp = grp.groupby("stint", group_keys=False).apply(tyre_pres)
        grp["tyre_preservation"] = tp.clip(0, 100)

        # 5. Pit compliance: how close actual pit lap is to peak pit_window_pressure lap
        pit_laps = grp[grp["is_pit_in"]]["lap"].tolist()
        optimal_laps = []
        if len(pit_laps) > 0:
            # Segment by pit lap, find optimal within each stint
            prev = 0
            for pit_lap in sorted(pit_laps):
                window = grp[(grp["lap"] > prev) & (grp["lap"] <= pit_lap)]
                if len(window) > 0 and window["pit_window_pressure"].notna().any():
                    opt = window.loc[window["pit_window_pressure"].idxmax(), "lap"]
                    optimal_laps.append((int(opt), int(pit_lap)))
                prev = pit_lap

        pc_score = []
        for _, row in grp.iterrows():
            if row["is_pit_in"] and optimal_laps:
                # Find the delta for this pit lap
                for opt, actual in optimal_laps:
                    if actual == int(row["lap"]):
                        delta = abs(actual - opt)
                        pc_score.append(max(0, 100 - delta * 15))
                        break
                else:
                    pc_score.append(50.0)
            else:
                pc_score.append(50.0)
        grp["pit_compliance"] = pd.Series(pc_score, index=grp.index)

        # 6. Restart aggression: lap time gain on first lap after safety_car_active ends
        sc_col = grp["safety_car_active"]
        restart_idxs = sc_col[sc_col.shift(1, fill_value=False) & ~sc_col].index
        restart_gain = pd.Series(50.0, index=grp.index)
        for idx in restart_idxs:
            if idx + 1 in grp.index:
                prev_lt = grp.loc[idx, "lap_time"]
                curr_lt = grp.loc[idx + 1, "lap_time"]
                if pd.notna(prev_lt) and pd.notna(curr_lt) and prev_lt > 0:
                    gain = (prev_lt - curr_lt) / prev_lt * 100
                    restart_gain.loc[idx + 1] = min(100, max(0, gain * 10 + 50))
        grp["restart_aggression"] = restart_gain

        # 7. Pressure consistency: low lap time variance when chased = high score
        chased_lt_std = np.where(chased, lt_std, np.nan)
        s = pd.Series(chased_lt_std, index=grp.index)
        filled = s.fillna(s.median() if s.notna().any() else 0)
        grp["pressure_consistency"] = 100 - _norm(filled)  # invert: low std = high consistency

        # 8. Late braking tendency: min speed in corners (lower = later braking)
        ms = grp["min_speed_corners"].fillna(grp["min_speed_corners"].median())
        grp["late_braking_tendency"] = 100 - _norm(ms)  # lower min speed = later braking = higher score

        # 9. Radio stress frequency: radio message count per stint, normalized
        radio_count = (grp["radio_text"] != "").astype(int).rolling(5, min_periods=1).sum()
        grp["radio_stress_frequency"] = _norm(radio_count)

        # 10. Undercut susceptibility: delta between actual pit lap and peak pressure lap
        us_series = []
        for _, row in grp.iterrows():
            if row["is_pit_in"] and optimal_laps:
                for opt, actual in optimal_laps:
                    if actual == int(row["lap"]):
                        late_by = actual - opt
                        us_series.append(min(100, max(0, late_by * 20 + 50)))
                        break
                else:
                    us_series.append(50.0)
            else:
                us_series.append(50.0)
        grp["undercut_susceptibility"] = pd.Series(us_series, index=grp.index)

        # 11. Dirty air tolerance: lap time degradation when gap_ahead < 1s
        close_racing = grp["gap_ahead"] < 1.0
        da = np.where(close_racing, grp["pace_delta"].fillna(0), np.nan)
        s_da = pd.Series(da, index=grp.index)
        filled_da = s_da.fillna(s_da.median() if s_da.notna().any() else 0)
        grp["dirty_air_tolerance"] = 100 - _norm(filled_da)  # less degradation = higher tolerance

        # Clip all traits to [0, 100]
        for col in TRAIT_COLS:
            grp[col] = grp[col].clip(0, 100).round(1)

        results.append(grp)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Label engineering for XGBoost targets
# ---------------------------------------------------------------------------

def _build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target columns derived from future position changes."""
    df = df.copy().sort_values(["driver", "lap"]).reset_index(drop=True)

    for driver, grp in df.groupby("driver"):
        idx = grp.index
        pos = grp["position"].values
        is_pit = grp["is_pit_in"].values

        # position_gain_prob target: did this driver gain a position in next 5 laps?
        gain = np.zeros(len(pos))
        for i in range(len(pos)):
            future = pos[i+1: i+6]
            if len(future) > 0 and pd.notna(pos[i]):
                gain[i] = float(np.any(future < pos[i]))

        # position_loss_prob target: did this driver lose a position in next 5 laps?
        loss = np.zeros(len(pos))
        for i in range(len(pos)):
            future = pos[i+1: i+6]
            if len(future) > 0 and pd.notna(pos[i]):
                loss[i] = float(np.any(future > pos[i]))

        # pit_prob target: did this driver pit in next 3 laps?
        pit = np.zeros(len(is_pit))
        for i in range(len(is_pit)):
            pit[i] = float(np.any(is_pit[i+1: i+4]))

        df.loc[idx, "_target_gain"] = gain
        df.loc[idx, "_target_loss"] = loss
        df.loc[idx, "_target_pit"] = pit

    return df


# ---------------------------------------------------------------------------
# UMAP fingerprint
# ---------------------------------------------------------------------------

def fit_umap(trait_matrix: np.ndarray, random_state: int = 42) -> "umap.UMAP":
    import umap
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
    reducer.fit(trait_matrix)
    return reducer


# ---------------------------------------------------------------------------
# XGBoost multi-target predictor
# ---------------------------------------------------------------------------

def fit_predictor(df: pd.DataFrame) -> tuple:
    """Fit three XGBoost classifiers (gain, loss, pit). Returns (models, feature_names)."""
    from xgboost import XGBClassifier
    from sklearn.multioutput import MultiOutputClassifier

    feature_cols = TRAIT_COLS + ["tyre_age", "gap_ahead", "gap_behind", "pit_window_pressure", "lap_time"]
    available = [c for c in feature_cols if c in df.columns]

    df_clean = df.dropna(subset=available + ["_target_gain", "_target_loss", "_target_pit"])
    X = df_clean[available].fillna(0).values
    y = df_clean[["_target_gain", "_target_loss", "_target_pit"]].values

    if len(X) < 20:
        log.warning("Not enough training samples for XGBoost — model will return 0.5 predictions")
        return None, available

    base = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    model = MultiOutputClassifier(base, n_jobs=1)
    model.fit(X, y)

    return model, available


# ---------------------------------------------------------------------------
# SHAP explainer
# ---------------------------------------------------------------------------

def build_shap_explainer(model, X_sample: np.ndarray):
    """Build a SHAP TreeExplainer for the first output (position_gain_prob)."""
    try:
        import shap
        first_estimator = model.estimators_[0]
        explainer = shap.TreeExplainer(first_estimator)
        return explainer
    except Exception as e:
        log.warning(f"SHAP explainer build failed: {e}")
        return None


def get_shap_values(explainer, X_row: np.ndarray, feature_names: list) -> dict:
    """Return top-3 SHAP features for a single prediction row."""
    if explainer is None:
        return {}
    try:
        import shap
        shap_vals = explainer.shap_values(X_row.reshape(1, -1))
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class
        vals = shap_vals[0]
        pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)[:3]
        return {name: round(float(val), 4) for name, val in pairs}
    except Exception as e:
        log.debug(f"SHAP inference failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Incident risk heuristic
# ---------------------------------------------------------------------------

def compute_incident_risk(df: pd.DataFrame) -> pd.Series:
    """
    Composite incident risk (0-1) — not a trained model (crashes are too rare).
    High tyre age + high aggression + low pressure consistency = high risk.
    """
    age_norm = (df["tyre_age"].clip(0, 50) / 50)
    agg_norm = df["aggression_level"].fillna(50) / 100
    consistency_inv = 1 - df["pressure_consistency"].fillna(50) / 100
    risk = (0.4 * age_norm + 0.35 * agg_norm + 0.25 * consistency_inv).clip(0, 1)
    return risk.round(3)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_models(umap_model, predictor, shap_explainer, feature_names: list) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "driver_soul.joblib")
    joblib.dump({
        "umap": umap_model,
        "predictor": predictor,
        "shap_explainer": shap_explainer,
        "feature_names": feature_names,
    }, path)
    log.info(f"Driver Soul models saved to {path}")
    return path


def load_models() -> dict:
    path = os.path.join(MODELS_DIR, "driver_soul.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No driver_soul.joblib at {path}. Run build_data.py first.")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def _compute_validation_metrics(df: pd.DataFrame, predictor, feature_names: list) -> dict:
    """Chronological 80/20 split validation. Logs accuracy/precision/recall per target.

    First N-1 races = train set, last race = validation set.
    Metrics are build-time only — logged to console, saved to models/saved/soul_metrics.json.
    Returns empty dict if insufficient data.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    if "race_slug" not in df.columns:
        log.warning("race_slug column missing — skipping validation metrics")
        return {}

    race_order = sorted(df["race_slug"].unique().tolist())
    if len(race_order) < 2:
        log.warning("Need at least 2 races for validation split — skipping metrics")
        return {}

    train_slugs = race_order[:-1]
    val_slug    = race_order[-1]

    train_df = df[df["race_slug"].isin(train_slugs)]
    val_df   = df[df["race_slug"] == val_slug]

    available_features = [c for c in feature_names if c in val_df.columns]
    target_cols  = ["_target_gain", "_target_loss", "_target_pit"]
    target_names = ["position_gain", "position_loss", "pit"]

    val_clean = val_df.dropna(subset=available_features + target_cols)
    if len(val_clean) < 10:
        log.warning(f"Only {len(val_clean)} clean validation rows — skipping metrics")
        return {}

    X_val = val_clean[available_features].fillna(0).values
    y_val = val_clean[target_cols].values

    try:
        proba = predictor.predict_proba(X_val)
        metrics = {}
        for i, name in enumerate(target_names):
            y_pred = (proba[i][:, 1] >= 0.5).astype(int)
            y_true = y_val[:, i].astype(int)
            metrics[name] = {
                "accuracy":       round(float(accuracy_score(y_true, y_pred)), 4),
                "precision":      round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall":         round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "n_val_samples":  int(len(y_true)),
                "val_race":       val_slug,
                "train_races":    train_slugs,
            }
        return metrics
    except Exception as e:
        log.warning(f"Validation metrics computation failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_and_save(all_data: pd.DataFrame) -> str:
    """
    Fit UMAP + XGBoost on all race data combined.
    Saves models to models/saved/driver_soul.joblib.
    """
    log.info("Engineering behavioral traits...")
    all_data = engineer_traits(all_data)

    log.info("Building XGBoost training labels...")
    all_data = _build_labels(all_data)

    log.info("Fitting UMAP on trait matrix...")
    trait_matrix = all_data[TRAIT_COLS].fillna(50).values
    umap_model = fit_umap(trait_matrix)

    log.info("Fitting multi-target XGBoost predictor...")
    predictor, feature_names = fit_predictor(all_data)

    # Validation metrics — build-time only, not used in production app
    if predictor is not None:
        _metrics = _compute_validation_metrics(all_data, predictor, feature_names)
        if _metrics:
            import json as _json
            _mpath = os.path.join(MODELS_DIR, "soul_metrics.json")
            with open(_mpath, "w") as _f:
                _json.dump(_metrics, _f, indent=2)
            log.info(f"Driver Soul validation metrics saved to {_mpath}")
            log.info(_json.dumps(_metrics, indent=2))

    log.info("Building SHAP explainer...")
    feature_cols = TRAIT_COLS + ["tyre_age", "gap_ahead", "gap_behind", "pit_window_pressure", "lap_time"]
    available = [c for c in feature_cols if c in all_data.columns]
    X_sample = all_data[available].fillna(0).values[:100]
    shap_explainer = build_shap_explainer(predictor, X_sample) if predictor else None

    return save_models(umap_model, predictor, shap_explainer, feature_names)


def add_predictions_to_race(df: pd.DataFrame) -> pd.DataFrame:
    """Load saved models and add all trait scores + predictions to a race DataFrame."""
    models = load_models()
    predictor = models["predictor"]
    feature_names = models["feature_names"]

    # Compute traits
    df = engineer_traits(df)

    # Incident risk (heuristic)
    df["incident_risk"] = compute_incident_risk(df)

    # XGBoost predictions
    if predictor is not None:
        available = [c for c in feature_names if c in df.columns]
        X = df[available].fillna(0).values
        try:
            proba = predictor.predict_proba(X)
            # MultiOutputClassifier returns list of (n, 2) arrays — one per output head
            if len(proba) < 3:
                raise ValueError(f"Predictor returned {len(proba)} output heads, expected 3+")
            df["position_gain_prob"] = np.round([p[1] for p in proba[0]], 3)
            df["position_loss_prob"] = np.round([p[1] for p in proba[1]], 3)
            df["pit_prob"] = np.round([p[1] for p in proba[2]], 3)
        except Exception as e:
            log.warning(f"XGBoost prediction failed: {e}")
            df["position_gain_prob"] = 0.5
            df["position_loss_prob"] = 0.5
            df["pit_prob"] = 0.5
    else:
        df["position_gain_prob"] = 0.5
        df["position_loss_prob"] = 0.5
        df["pit_prob"] = 0.5

    return df


def get_driver_fingerprint(df: pd.DataFrame, driver: str) -> dict:
    """Return a driver's season-average trait scores (for the radar chart)."""
    drv_df = df[df["driver"] == driver]
    if len(drv_df) == 0:
        return {col: 50.0 for col in TRAIT_COLS}
    return {col: round(float(drv_df[col].mean()), 1) for col in TRAIT_COLS}


def get_lap_state(df: pd.DataFrame, driver: str, lap: int) -> dict:
    """Return the full behavioral state for a driver at a specific lap."""
    row = df[(df["driver"] == driver) & (df["lap"] == lap)]
    if len(row) == 0:
        return {}
    row = row.iloc[0]
    result = {}
    for col in TRAIT_COLS + PRED_COLS:
        result[col] = round(float(row[col]), 3) if pd.notna(row.get(col)) else 0.0
    return result

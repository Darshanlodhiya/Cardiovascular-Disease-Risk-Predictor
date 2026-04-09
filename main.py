from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import sys
import shap
import numpy as np
import joblib
import pandas as pd
import __main__
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

app = FastAPI()

# threshold model
class ThresholdModel:
    def __init__(self, model=None, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Some serialized models reference `__main__.ThresholdModel` (from the training script).
# When running under Uvicorn, `__main__` is not this file, so we must register it explicitly.
setattr(__main__, "ThresholdModel", ThresholdModel)
sys.modules["__main__"].ThresholdModel = ThresholdModel

# ==============================
# LOAD ALL MODELS (REGISTRY)
# ==============================
MODEL_REGISTRY = {}

def load_model(path):
    try:
        # Ensure custom wrapper is available during unpickling
        setattr(__main__, "ThresholdModel", ThresholdModel)
        sys.modules["__main__"].ThresholdModel = ThresholdModel
        m = joblib.load(path)
        if hasattr(m, "model"):
            m = m.model
        return m
    except ModuleNotFoundError as e:
        # Some pickles require optional libs (e.g., lightgbm, xgboost).
        # Keep API bootable even if a specific model can't be loaded.
        print(f"Model '{path}' could not be loaded (missing dependency): {e}")
        return None
    except Exception as e:
        print(f"Model '{path}' could not be loaded: {e}")
        return None

MODEL_REGISTRY = {
    "tuned_ensemble2.pkl": load_model("tuned_ensemble2.pkl"),
    "tuned_rf.pkl": load_model("tuned_rf.pkl"),
    "tuned_xgb.pkl": load_model("tuned_xgb.pkl"),
    "tuned_ensemble.pkl": load_model("tuned_ensemble.pkl"),
    "tuned_lgbm.pkl": load_model("tuned_lgbm.pkl"),
    "tuned_dt.pkl": load_model("tuned_dt.pkl"),
}


# default fallback model
model = MODEL_REGISTRY.get("tuned_ensemble2.pkl") or next(
    (m for m in MODEL_REGISTRY.values() if m is not None),
    None,
)

# ==============================
# SHAP EXPLAINERS
# ==============================
EXPLAINER_REGISTRY = {}
_BACKGROUND_REGISTRY: dict[str, pd.DataFrame] = {}

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

def _get_or_build_tree_explainer(model_key: str, selected_model):
    """
    Build/caches a TreeExplainer when the model is supported.
    For unsupported models, returns None (caller may use a fallback).
    """
    cached = EXPLAINER_REGISTRY.get(model_key)
    if cached is not None:
        return cached

    try:
        explainer = shap.TreeExplainer(selected_model)
        EXPLAINER_REGISTRY[model_key] = explainer
        return explainer
    except Exception as e:
        print(f"TreeExplainer unavailable for {model_key}: {e}")
        return None


def _compute_shap_values_and_base(model_key: str, selected_model, input_df: pd.DataFrame) -> tuple[dict, float]:
    """
    Returns ({feature: shap_value}, base_value) for the positive class when possible.
    base_value is SHAP's expected value for the same output (typically log-odds).
    """
    # 1) Preferred: TreeExplainer (fast, accurate for tree models)
    explainer = _get_or_build_tree_explainer(model_key, selected_model)
    if explainer is not None:
        shap_values = explainer.shap_values(input_df)

        # classification can return list[class_index] arrays
        if isinstance(shap_values, list):
            arr = np.asarray(shap_values[1])
            # expected: (n_samples, n_features)
            # sometimes: (n_samples, n_features, n_outputs)
            if arr.ndim == 3 and arr.shape[-1] >= 2:
                values_1d = arr[0, :, 1]
            elif arr.ndim == 2:
                values_1d = arr[0, :]
            else:
                values_1d = arr.reshape(-1)
        else:
            arr = np.asarray(shap_values)
            # can be (n_samples, n_features) or (n_samples, n_features, n_outputs)
            if arr.ndim == 3 and arr.shape[-1] >= 2:
                values_1d = arr[0, :, 1]
            elif arr.ndim == 2:
                values_1d = arr[0, :]
            else:
                values_1d = arr.reshape(-1)

        values_1d = np.asarray(values_1d, dtype=float)

        # Some wrapped/ensemble models may allow TreeExplainer construction but yield
        # effectively-all-zero SHAP values. If that happens, fall back to a robust
        # model-agnostic explainer so the UI shows meaningful attributions.
        if np.all(~np.isfinite(values_1d)) or np.allclose(values_1d, 0.0, atol=1e-10):
            explainer = None  # trigger fallback below
        else:
            expected = getattr(explainer, "expected_value", 0.0)
            if isinstance(expected, (list, tuple, np.ndarray)):
                base_value = float(np.asarray(expected).reshape(-1)[1] if len(expected) > 1 else np.asarray(expected).reshape(-1)[0])
            else:
                base_value = float(expected)

            shap_values_dict = {feature: float(val) for feature, val in zip(input_df.columns, values_1d)}
            return shap_values_dict, base_value

    # 2) Fallback: model-agnostic explainer on a tiny background (1 row)
    # NOTE: This avoids "no SHAP" for ensembles/wrappers, but can be slower.
    # IMPORTANT: background must not be identical to the input row, otherwise permutation SHAP can collapse to all zeros.
    background = _BACKGROUND_REGISTRY.get(model_key)
    if background is None:
        try:
            # Build a small background set from the training dataset, aligned to the model's features.
            df_bg_raw = pd.read_csv("CVD_Dataset.csv").sample(n=200, random_state=42)
            if "CVD_Risk" in df_bg_raw.columns:
                df_bg_raw = df_bg_raw.drop(columns=["CVD_Risk"])
            background = _preprocess_dataset_frame(df_bg_raw, selected_model=selected_model)
        except Exception:
            # Last-resort: use the input row duplicated (may yield weak explanations).
            background = input_df.copy()
        _BACKGROUND_REGISTRY[model_key] = background

    masker = shap.maskers.Independent(background)

    def _predict_proba(X):
        X_df = pd.DataFrame(X, columns=input_df.columns)
        try:
            return selected_model.predict_proba(X_df)
        except Exception:
            preds = selected_model.predict(X_df)
            preds = np.asarray(preds).reshape(-1, 1)
            return np.hstack([1 - preds, preds])

    explainer = shap.Explainer(_predict_proba, masker, algorithm="permutation")
    explanation = explainer(input_df)

    # explanation.values shape: (n_samples, n_features, n_outputs) for proba
    values = np.asarray(explanation.values)
    if values.ndim == 3 and values.shape[-1] >= 2:
        values_1d = values[0, :, 1]
    elif values.ndim == 2:
        values_1d = values[0, :]
    else:
        values_1d = values.reshape(-1)

    base_values = np.asarray(getattr(explanation, "base_values", 0.0))
    if base_values.ndim >= 2:
        base_value = float(base_values[0, 1] if base_values.shape[-1] >= 2 else base_values.reshape(-1)[0])
    else:
        base_value = float(base_values.reshape(-1)[0] if base_values.size else 0.0)

    shap_values_dict = {feature: float(val) for feature, val in zip(input_df.columns, values_1d)}
    return shap_values_dict, base_value


def _shap_probability_breakdown(
    shap_values: dict,
    base_value: float,
    predicted_proba: float,
    top_k: int = 10,
) -> list[dict]:
    """
    Produces a signed breakdown in *probability percentage points* that sums to
    (predicted_proba - base_proba).

    This keeps the core SHAP semantics: baseline + per-feature signed effects -> final prediction,
    but renders it in % points for non-technical users.
    """
    p_base = _sigmoid(float(base_value))
    target_delta = float(predicted_proba) - float(p_base)  # probability units (0..1)

    # Use raw SHAP values (typically log-odds) for stable attribution.
    # SHAP guarantees: base_value + sum(shap_values) ~= model_output (in the same space).
    items = [(f, float(sv)) for f, sv in shap_values.items() if np.isfinite(sv)]
    if not items:
        return []

    items.sort(key=lambda x: abs(x[1]), reverse=True)
    use = items[: max(1, int(top_k))]

    denom = float(sum(v for _, v in items))  # use ALL features to avoid near-zero denom from top-k cancellation
    if abs(denom) < 1e-12:
        # If prediction ~= baseline, deltas are inherently tiny; still return top-k with proportional zeros.
        return [{"feature": f, "pct_points": 0.0} for f, _ in use]

    scale = float((target_delta * 100.0) / denom)  # convert to percentage points per SHAP unit
    breakdown = [{"feature": f, "pct_points": float(v * scale)} for f, v in use]

    # Add an "Other" bucket so the displayed breakdown sums exactly to (pred - base)
    shown_sum = float(sum(b["pct_points"] for b in breakdown))
    target_pp = float(target_delta * 100.0)
    other_pp = float(target_pp - shown_sum)
    if abs(other_pp) > 1e-10:
        breakdown.append({"feature": "Other", "pct_points": other_pp})

    return breakdown


def _priority_from_impact(impact_pp: float) -> str:
    if impact_pp >= 8:
        return "High"
    if impact_pp >= 3:
        return "Medium"
    return "Low"


def _safety_flags(user_input: "UserInput") -> list[str]:
    flags: list[str] = []
    if user_input.Blood_Pressure_Systolic >= 180 or user_input.Blood_Pressure_Diastolic >= 120:
        flags.append("Very high blood pressure range. Seek urgent medical assessment.")
    if user_input.Fasting_Blood_Sugar >= 200:
        flags.append("Very high fasting glucose. Prompt clinician review is advised.")
    if user_input.HbA1c >= 9.0:
        flags.append("Very high HbA1c. Diabetes treatment review is recommended soon.")
    if user_input.LDL >= 190:
        flags.append("LDL is very high (>=190 mg/dL). Medical lipid management should be discussed.")
    return flags


def _build_recommendation_for_feature(feature: str, user_input: "UserInput", impact_pp: float) -> dict | None:
    # Build one recommendation per modifiable SHAP feature to keep order aligned.
    rec: dict | None = None
    safety_flag: str | None = None

    if feature in {"Blood_Pressure_Systolic", "Blood_Pressure_Diastolic"}:
        rec = {
            "feature": feature,
            "current_value": f"{user_input.Blood_Pressure_Systolic}/{user_input.Blood_Pressure_Diastolic} mmHg",
            "target": "Home BP consistently <130/80 mmHg",
            "action": "Measure BP at home for 7 days, reduce salt to <5 g/day, and do brisk walking 30 minutes on most days.",
        }
        if user_input.Blood_Pressure_Systolic >= 180 or user_input.Blood_Pressure_Diastolic >= 120:
            safety_flag = "Urgent medical assessment advised for very high blood pressure."
    elif feature == "Pulse_Pressure":
        rec = {
            "feature": feature,
            "current_value": f"{user_input.Blood_Pressure_Systolic - user_input.Blood_Pressure_Diastolic} mmHg",
            "target": "Narrow pulse pressure trend via BP control",
            "action": "Control systolic BP with daily activity, low-salt meals, and medication review if readings stay high.",
        }
    elif feature == "Hypertension":
        rec = {
            "feature": feature,
            "current_value": "Yes" if user_input.Hypertension == 1 else "No",
            "target": "Sustained BP control and adherence",
            "action": "Do not miss BP medicines, track BP weekly, and review treatment if average home BP remains above target.",
        }
    elif feature in {"LDL", "Total_Cholesterol", "Triglycerides", "HDL"}:
        rec = {
            "feature": feature,
            "current_value": (
                f"LDL {user_input.LDL}, Total {user_input.Total_Cholesterol}, "
                f"TG {user_input.Triglycerides}, HDL {user_input.HDL} mg/dL"
            ),
            "target": "LDL <100 mg/dL, TG <150 mg/dL, HDL at least 40-50 mg/dL",
            "action": "Replace fried/processed foods with fiber-rich meals, add weekly exercise, and repeat lipid tests in 8-12 weeks.",
        }
        if user_input.LDL >= 190:
            safety_flag = "LDL is very high; discuss early lipid-lowering treatment with your clinician."
    elif feature in {"Fasting_Blood_Sugar", "HbA1c"}:
        rec = {
            "feature": feature,
            "current_value": f"FBS {user_input.Fasting_Blood_Sugar} mg/dL, HbA1c {user_input.HbA1c}%",
            "target": "Fasting sugar 80-130 mg/dL and HbA1c <7% (individualized)",
            "action": "Cut refined carbs/sugary drinks, walk 10-15 minutes after meals, and monitor sugars as advised.",
        }
        if user_input.Fasting_Blood_Sugar >= 200:
            safety_flag = "Very high fasting glucose needs prompt medical review."
        if user_input.HbA1c >= 9.0:
            safety_flag = "Very high HbA1c needs treatment review soon."
    elif feature == "Diabetes":
        rec = {
            "feature": feature,
            "current_value": "Yes" if user_input.Diabetes == 1 else "No",
            "target": "Consistent glucose control and routine follow-up",
            "action": "Follow medication and meal timing regularly, and schedule periodic HbA1c and kidney/eye screening.",
        }
    elif feature in {"BMI", "Metabolic Syndrome", "Cardio_Risk_Score"}:
        rec = {
            "feature": feature,
            "current_value": f"BMI {user_input.BMI}",
            "target": "Reduce body weight by 5-10% over 6 months if overweight",
            "action": "Use plate-based portion control, keep a daily step goal, and include strength training 2-3 times/week.",
        }
    elif feature in {"Smoking_Status"}:
        rec = {
            "feature": feature,
            "current_value": user_input.Smoking_Status,
            "target": "Complete tobacco cessation",
            "action": "Set a quit date, avoid trigger situations, and use cessation counseling/therapy support.",
        }
    elif feature in {"Alcohol_Consumption"}:
        rec = {
            "feature": feature,
            "current_value": user_input.Alcohol_Consumption,
            "target": "Avoid alcohol or keep to minimal intake",
            "action": "Plan alcohol-free days each week and avoid binge drinking.",
        }
    elif feature in {"Daily_Steps"}:
        rec = {
            "feature": feature,
            "current_value": f"{user_input.Daily_Steps} steps/day",
            "target": "Increase by 1000-2000 steps/day in the next 2-4 weeks",
            "action": "Add two 10-15 minute brisk walks (after meals works well).",
        }
    elif feature in {"Physical_Activity_Level"}:
        rec = {
            "feature": feature,
            "current_value": user_input.Physical_Activity_Level,
            "target": "At least 150-300 min/week moderate activity",
            "action": "Schedule 30 minutes of moderate activity on 5 days/week plus 2 days of strength work.",
        }
    elif feature in {"Sleep_Hours", "Sleep_Risk"}:
        rec = {
            "feature": feature,
            "current_value": f"{user_input.Sleep_Hours} hours/night",
            "target": "7-9 hours sleep with a regular schedule",
            "action": "Fix bedtime/wake time daily and avoid caffeine/screens in the late evening.",
        }
    elif feature in {"Screen_Time_Hours", "High_Screen_Time"}:
        rec = {
            "feature": feature,
            "current_value": f"{user_input.Screen_Time_Hours} hours/day",
            "target": "Reduce recreational screen exposure, especially before sleep",
            "action": "Set app timers and keep the last 60 minutes before bed screen-free.",
        }
    elif feature in {"Stress_Level", "Work_Stress_Type_Shift-based", "Work_Stress_Type_Sedentary"}:
        rec = {
            "feature": feature,
            "current_value": f"Stress {user_input.Stress_Level}/10, work pattern {user_input.Work_Stress_Type}",
            "target": "Lower day-to-day stress burden",
            "action": "Practice 10 minutes daily breathing/mindfulness and add short movement breaks every 60 minutes at work.",
        }
    elif feature in {"Junk_Food_Frequency"}:
        rec = {
            "feature": feature,
            "current_value": f"{user_input.Junk_Food_Frequency}/10",
            "target": "Limit junk food to <=1-2 times/week",
            "action": "Pre-plan healthy snacks and swap one processed meal each day with a home-cooked whole-food option.",
        }
    elif feature in {"Diet_Type_Non-Vegetarian"}:
        rec = {
            "feature": feature,
            "current_value": user_input.Diet_Type,
            "target": "Heart-healthy meal pattern (DASH-style)",
            "action": "Prioritize vegetables, pulses, whole grains, and lean proteins while reducing red/processed meat portions.",
        }
    elif feature in {"Air_Pollution_Exposure"}:
        rec = {
            "feature": feature,
            "current_value": user_input.Air_Pollution_Exposure,
            "target": "Reduce peak pollution exposure where possible",
            "action": "Avoid outdoor exercise during high AQI periods and use cleaner commuting/indoor air strategies.",
        }

    if rec is None:
        return None

    rec["priority"] = _priority_from_impact(impact_pp)
    rec["impact"] = float(impact_pp)
    rec["safety_flag"] = safety_flag
    return rec


def _generate_recommendations(user_input: "UserInput", shap_breakdown: list[dict]) -> list[dict]:
    # Keep same order as "What increased your risk" list in SHAP breakdown.
    positive_items = [
        item for item in shap_breakdown
        if item.get("feature") not in {None, "Other"} and float(item.get("pct_points", 0.0)) > 0
    ]

    recommendations: list[dict] = []
    for item in positive_items:
        feature = str(item.get("feature"))
        impact_pp = float(item.get("pct_points", 0.0))
        rec = _build_recommendation_for_feature(feature=feature, user_input=user_input, impact_pp=impact_pp)
        if rec:
            recommendations.append(rec)

    # If no modifiable factor was captured, return empty list.
    if not recommendations:
        return []

    # Attach global safety flags once, to the first recommendation, without changing order.
    flags = _safety_flags(user_input)
    if flags:
        merged = " | ".join(flags)
        recommendations[0]["safety_flag"] = (
            f"{recommendations[0]['safety_flag']} | {merged}"
            if recommendations[0].get("safety_flag")
            else merged
        )

    return recommendations


# encoding dictionaries
GENDER_ENCODING = {'Male': 1, 'Female': 0, 'Other': 2}
REGION_ENCODING = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Central': 4}
URBAN_RURAL_ENCODING = {'Urban': 0, 'Rural': 1}
SOCIOECONOMIC_ENCODING = {'Lower': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'Upper': 4}
OCCUPATION_ENCODING = {'Unemployed': 0, 'Small_Business': 1, 'Labor_Worker': 2, 'IT_Professional': 3, 'Students': 4}
DIET_ENCODING = {'Vegetarian': 0, 'Non-Vegetarian': 1, 'Eggetarian': 2}
ACTIVITY_ENCODING = {'Low': 0, 'Moderate': 1, 'High': 2}
SMOKING_ENCODING = {'Never': 0, 'Occasional': 1, 'Regular': 2}
ALCOHOL_ENCODING = {'Low': 0, 'Moderate': 1, 'High': 2}
POLLUTION_ENCODING = {'Low': 0, 'Moderate': 1, 'High': 2}
WORK_STRESS_ENCODING = {'Sedentary': 0, 'Physical': 1, 'Shift-based': 2}

# ==============================
# DATASET / METRICS (for model comparison)
# ==============================
_MODEL_COMPARISON_CACHE: dict | None = None

def _preprocess_dataset_frame(df_raw: pd.DataFrame, selected_model=None) -> pd.DataFrame:
    """
    Vectorized preprocessing for evaluation using the same feature engineering as `preprocess_input`.
    Expects raw columns as in CVD_Dataset.csv (including CVD_Risk).
    """
    df = df_raw.copy()

    # Ensure required computed features exist (recompute to stay consistent with API)
    if "Pulse_Pressure" not in df.columns:
        df["Pulse_Pressure"] = df["Blood_Pressure_Systolic"] - df["Blood_Pressure_Diastolic"]
    if "Sleep_Risk" not in df.columns:
        df["Sleep_Risk"] = ((df["Sleep_Hours"] < 6) | (df["Sleep_Hours"] > 9)).astype(int)
    if "High_Screen_Time" not in df.columns:
        df["High_Screen_Time"] = (df["Screen_Time_Hours"] > 8).astype(int)
    if "Cardio_Risk_Score" not in df.columns:
        df["Cardio_Risk_Score"] = (
            (df["Total_Cholesterol"] > 200).astype(int)
            + (df["Triglycerides"] > 150).astype(int)
            + (df["Blood_Pressure_Systolic"] > 130).astype(int)
            + (df["BMI"] > 25).astype(int)
            + (df["Diabetes"] == 1).astype(int)
            + (df["Hypertension"] == 1).astype(int)
            + (df["Smoking_Status"].isin(["Occasional", "Regular"])).astype(int)
            + (df["Family_History_CVD"] == 1).astype(int)
        )
    # Column in CSV is "Metabolic Syndrome" (with space)
    if "Metabolic Syndrome" not in df.columns:
        metabolic_score = (
            (df["BMI"] >= 25).astype(int)
            + (df["Diabetes"] == 1).astype(int)
            + (df["Triglycerides"] >= 150).astype(int)
            + ((df["Blood_Pressure_Systolic"] >= 130) | (df["Blood_Pressure_Diastolic"] >= 85)).astype(int)
        )
        df["Metabolic Syndrome"] = (metabolic_score >= 3).astype(int)

    # Label encodings (match API)
    df["Air_Pollution_Exposure"] = df["Air_Pollution_Exposure"].map(POLLUTION_ENCODING).astype(int)
    df["Urban_Rural"] = df["Urban_Rural"].map(URBAN_RURAL_ENCODING).astype(int)
    df["Socioeconomic_Status"] = df["Socioeconomic_Status"].map(SOCIOECONOMIC_ENCODING).astype(int)
    df["Physical_Activity_Level"] = df["Physical_Activity_Level"].map(ACTIVITY_ENCODING).astype(int)
    df["Smoking_Status"] = df["Smoking_Status"].map(SMOKING_ENCODING).astype(int)
    df["Alcohol_Consumption"] = df["Alcohol_Consumption"].map(ALCOHOL_ENCODING).astype(int)

    # One-hot encodings (match API)
    df["Gender_Male"] = (df["Gender"] == "Male").astype(int)
    df["Gender_Other"] = (df["Gender"] == "Other").astype(int)

    for region in ["East", "North", "South", "West"]:
        df[f"Region_{region}"] = (df["Region"] == region).astype(int)

    diet = df["Diet_Type"].replace({"Eggetarian": "Vegetarian"})
    df["Diet_Type_Non-Vegetarian"] = (diet == "Non-Vegetarian").astype(int)
    df["Diet_Type_Vegetarian"] = (diet == "Vegetarian").astype(int)

    for w in ["Sedentary", "Shift-based"]:
        df[f"Work_Stress_Type_{w}"] = (df["Work_Stress_Type"] == w).astype(int)

    for occ in ["Labor_Worker", "Small_Business", "Students", "Unemployed"]:
        df[f"Occupation_{occ}"] = (df["Occupation"] == occ).astype(int)

    # Drop raw categoricals that models typically don't use directly
    for col in ["Gender", "Region", "Diet_Type", "Work_Stress_Type", "Occupation"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Align with model features
    if selected_model is not None and hasattr(selected_model, "feature_names_in_"):
        for col in selected_model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[selected_model.feature_names_in_]

    return df


def _evaluate_model_on_dataset(model_key: str, selected_model, df_raw: pd.DataFrame) -> dict:
    y = df_raw["CVD_Risk"].astype(int).to_numpy()
    X = _preprocess_dataset_frame(df_raw.drop(columns=["CVD_Risk"]), selected_model=selected_model)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Models are already trained; we only evaluate on the held-out split
    y_pred = selected_model.predict(X_test)

    try:
        y_proba = selected_model.predict_proba(X_test)[:, 1]
    except Exception:
        # Fallback if model doesn't support predict_proba
        y_proba = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    # roc_auc requires both classes present; stratify should ensure this
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = None

    return {
        "model_key": model_key,
        "n_rows": int(len(df_raw)),
        "test_size": 0.2,
        "random_state": 42,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _compute_model_comparison(force: bool = False) -> dict:
    global _MODEL_COMPARISON_CACHE
    if _MODEL_COMPARISON_CACHE is not None and not force:
        return _MODEL_COMPARISON_CACHE

    dataset_path = "CVD_Dataset.csv"
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not read dataset '{dataset_path}': {e}")

    if "CVD_Risk" not in df.columns:
        raise HTTPException(status_code=503, detail="Dataset missing required target column 'CVD_Risk'")

    results = []
    for model_key, m in MODEL_REGISTRY.items():
        if m is None:
            results.append({"model_key": model_key, "available": False, "error": "Model unavailable (missing dependency or failed to load)."})
            continue
        try:
            metrics = _evaluate_model_on_dataset(model_key, m, df)
            metrics["available"] = True
            results.append(metrics)
        except Exception as e:
            results.append({"model_key": model_key, "available": False, "error": str(e)})

    payload = {
        "dataset": dataset_path,
        "target": "CVD_Risk",
        "models": results,
    }
    _MODEL_COMPARISON_CACHE = payload
    return payload

# Pydantic model for user input - aligned with CVD_Dataset.csv features
class UserInput(BaseModel):
    
    # Demographic Information
    Age: Annotated[int, Field(..., ge=18, le=100, description='Age of the person in years', example=25)]
    Gender: Annotated[Literal['Male', 'Female', 'Other'], Field(..., description='Gender', example='Male')]
    Region: Annotated[Literal['North', 'South', 'East', 'West', 'Central'], Field(..., description='Geographic region', example='North')]
    Urban_Rural: Annotated[Literal['Urban', 'Rural'], Field(..., description='Urban or Rural area', example='Urban')]
    Socioeconomic_Status: Annotated[Literal['Lower', 'Lower-Middle', 'Middle', 'Upper-Middle', 'Upper'], Field(..., description='Socioeconomic status', example='Middle')]
    Occupation: Annotated[Literal['Unemployed', 'Small_Business', 'Labor_Worker', 'IT_Professional', 'Students'], Field(..., description='Type of occupation', example='IT_Professional')]
    
    # Lifestyle Information
    Diet_Type: Annotated[Literal['Vegetarian', 'Non-Vegetarian', 'Eggetarian'], Field(..., description='Diet preference', example='Vegetarian')]
    Junk_Food_Frequency: Annotated[int, Field(..., ge=0, le=10, description='Frequency of junk food consumption (0-10 scale)', example=5)]
    Physical_Activity_Level: Annotated[Literal['Low', 'Moderate', 'High'], Field(..., description='Physical activity level', example='Moderate')]
    Daily_Steps: Annotated[int, Field(..., ge=0, le=50000, description='Average daily steps walked', example=5000)]
    Screen_Time_Hours: Annotated[float, Field(..., ge=0, le=24, description='Daily screen time in hours', example=5.5)]
    Sleep_Hours: Annotated[float, Field(..., ge=0, le=24, description='Average daily sleep in hours', example=7.0)]
    Stress_Level: Annotated[int, Field(..., ge=1, le=10, description='Stress level on a scale of 1-10', example=5)]
    
    # Smoking and Alcohol
    Smoking_Status: Annotated[Literal['Never', 'Occasional', 'Regular'], Field(..., description='Smoking status', example='Never')]
    Alcohol_Consumption: Annotated[Literal['Low', 'Moderate', 'High'], Field(..., description='Alcohol consumption level', example='Moderate')]
    
    # Medical History
    Family_History_CVD: Annotated[int, Field(..., ge=0, le=1, description='Family history of CVD (1=Yes, 0=No)', example=0)]
    Diabetes: Annotated[int, Field(..., ge=0, le=1, description='Diabetes status (1=Yes, 0=No)', example=0)]
    Hypertension: Annotated[int, Field(..., ge=0, le=1, description='Hypertension status (1=Yes, 0=No)', example=0)]
    PCOS: Annotated[int, Field(..., ge=0, le=1, description='PCOS status (1=Yes, 0=No)', example=0)]
    
    # Physical Measurements
    BMI: Annotated[float, Field(..., ge=10, le=60, description='Body Mass Index', example=24.5)]
    Blood_Pressure_Systolic: Annotated[int, Field(..., ge=80, le=200, description='Systolic blood pressure (mmHg)', example=120)]
    Blood_Pressure_Diastolic: Annotated[int, Field(..., ge=40, le=130, description='Diastolic blood pressure (mmHg)', example=80)]
    Resting_Heart_Rate: Annotated[int, Field(..., ge=40, le=150, description='Resting heart rate (bpm)', example=70)]
    
    # Blood Markers
    Fasting_Blood_Sugar: Annotated[float, Field(..., ge=50, le=300, description='Fasting blood sugar (mg/dL)', example=100)]
    HbA1c: Annotated[float, Field(..., ge=3, le=15, description='HbA1c level (%)', example=5.5)]
    Total_Cholesterol: Annotated[float, Field(..., ge=50, le=400, description='Total cholesterol (mg/dL)', example=200)]
    LDL: Annotated[float, Field(..., ge=20, le=300, description='LDL cholesterol (mg/dL)', example=130)]
    HDL: Annotated[float, Field(..., ge=10, le=150, description='HDL cholesterol (mg/dL)', example=50)]
    Triglycerides: Annotated[float, Field(..., ge=20, le=500, description='Triglycerides (mg/dL)', example=150)]
    
    # Environmental Factors
    Air_Pollution_Exposure: Annotated[Literal['Low', 'Moderate', 'High'], Field(..., description='Level of air pollution exposure', example='Moderate')]
    Work_Stress_Type: Annotated[Literal['Sedentary', 'Physical', 'Shift-based'], Field(..., description='Type of work stress', example='Sedentary')]
    model_name: str | None = None

def preprocess_input(user_data: UserInput, selected_model=None) -> pd.DataFrame:

    # ===== Base numeric + label encoded =====
    data_dict = {
        'Age': user_data.Age,
        'Junk_Food_Frequency': user_data.Junk_Food_Frequency,
        'Daily_Steps': user_data.Daily_Steps,
        'Screen_Time_Hours': user_data.Screen_Time_Hours,
        'Sleep_Hours': user_data.Sleep_Hours,
        'Stress_Level': user_data.Stress_Level,
        'Family_History_CVD': user_data.Family_History_CVD,
        'Diabetes': user_data.Diabetes,
        'Hypertension': user_data.Hypertension,
        'PCOS': user_data.PCOS,
        'BMI': user_data.BMI,
        'Blood_Pressure_Systolic': user_data.Blood_Pressure_Systolic,
        'Blood_Pressure_Diastolic': user_data.Blood_Pressure_Diastolic,
        'Resting_Heart_Rate': user_data.Resting_Heart_Rate,
        'Fasting_Blood_Sugar': user_data.Fasting_Blood_Sugar,
        'HbA1c': user_data.HbA1c,
        'Total_Cholesterol': user_data.Total_Cholesterol,
        'LDL': user_data.LDL,
        'HDL': user_data.HDL,
        'Triglycerides': user_data.Triglycerides,

        # label encoding
        'Air_Pollution_Exposure': POLLUTION_ENCODING[user_data.Air_Pollution_Exposure],
        'Urban_Rural': URBAN_RURAL_ENCODING[user_data.Urban_Rural],
        'Socioeconomic_Status': SOCIOECONOMIC_ENCODING[user_data.Socioeconomic_Status],
        'Physical_Activity_Level': ACTIVITY_ENCODING[user_data.Physical_Activity_Level],
        'Smoking_Status': SMOKING_ENCODING[user_data.Smoking_Status],
        'Alcohol_Consumption': ALCOHOL_ENCODING[user_data.Alcohol_Consumption],
    }

    df = pd.DataFrame([data_dict])

    # ===== One-hot encoding =====

    # Gender
    df['Gender_Male'] = int(user_data.Gender == 'Male')
    df['Gender_Other'] = int(user_data.Gender == 'Other')

    # Region
    for region in ['East', 'North', 'South', 'West']:
        df[f'Region_{region}'] = int(user_data.Region == region)

    # Diet (Eggetarian treated as Vegetarian)
    diet = user_data.Diet_Type.replace('Eggetarian', 'Vegetarian')
    for d in ['Non-Vegetarian', 'Vegetarian']:
        df[f'Diet_Type_{d}'] = int(diet == d)

    # Work stress
    for w in ['Sedentary', 'Shift-based']:
        df[f'Work_Stress_Type_{w}'] = int(user_data.Work_Stress_Type == w)

    # Occupation
    for occ in ['Labor_Worker', 'Small_Business', 'Students', 'Unemployed']:
        df[f'Occupation_{occ}'] = int(user_data.Occupation == occ)

    # ===== Computed Features (FIXED) =====

    metabolic_score = (
        int(user_data.BMI >= 25) +
        int(user_data.Diabetes == 1) +
        int(user_data.Triglycerides >= 150) +
        int((user_data.Blood_Pressure_Systolic >= 130) or (user_data.Blood_Pressure_Diastolic >= 85))
    )
    df['Metabolic Syndrome'] = int(metabolic_score >= 3)

    df['Cardio_Risk_Score'] = (
        int(user_data.Total_Cholesterol > 200) +
        int(user_data.Triglycerides > 150) +
        int(user_data.Blood_Pressure_Systolic > 130) +
        int(user_data.BMI > 25) +
        int(user_data.Diabetes == 1) +
        int(user_data.Hypertension == 1) +
        int(user_data.Smoking_Status in ['Occasional', 'Regular']) +
        int(user_data.Family_History_CVD == 1)
    )

    df['Pulse_Pressure'] = (
        user_data.Blood_Pressure_Systolic - user_data.Blood_Pressure_Diastolic
    )

    df['Sleep_Risk'] = int(
        user_data.Sleep_Hours < 6 or user_data.Sleep_Hours > 9
    )

    df['High_Screen_Time'] = int(
        user_data.Screen_Time_Hours > 8
    )

    # ===== Align with model features =====
    if selected_model and hasattr(selected_model, 'feature_names_in_'):
        for col in selected_model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[selected_model.feature_names_in_]

    return df  

@app.post("/predict")
async def predict_cvd_risk(user_input: UserInput):

    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # SELECT MODEL
    model_key = user_input.model_name or "tuned_ensemble2.pkl"
    selected_model = MODEL_REGISTRY.get(model_key) or model
    if selected_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Selected model '{model_key}' is unavailable (missing dependency or failed to load).",
        )

    # preprocess
    input_df = preprocess_input(user_input, selected_model)

    # prediction
    prediction = selected_model.predict(input_df)[0]

    # probability
    try:
        proba = selected_model.predict_proba(input_df)[0][1]
    except:
        proba = float(prediction)

    risk_percent = round(proba * 100, 2)

    if risk_percent < 20:
        level = "Low"
    elif risk_percent < 50:
        level = "Moderate"
    elif risk_percent < 75:
        level = "High"
    else:
        level = "Very High"

    # ==============================
    # SHAP EXPLANATION (CORRECT PLACE)
    # ==============================
    try:
        shap_values_dict, shap_base_value = _compute_shap_values_and_base(
            model_key=model_key,
            selected_model=selected_model,
            input_df=input_df,
        )
        shap_base_proba = _sigmoid(float(shap_base_value))
        shap_breakdown = _shap_probability_breakdown(
            shap_values=shap_values_dict,
            base_value=shap_base_value,
            predicted_proba=float(proba),
            top_k=10,
        )
    except Exception as e:
        shap_values_dict = {"error": str(e)}
        shap_base_proba = None
        shap_breakdown = []
    recommendations = _generate_recommendations(user_input=user_input, shap_breakdown=shap_breakdown)

    return {
        "CVD_Risk": int(prediction),
        "risk_probability": float(risk_percent),
        "risk_probability_text": f"{risk_percent}%",
        "risk_level": level,
        "shap_values": shap_values_dict,
        "shap_base_probability": None if shap_base_proba is None else round(float(shap_base_proba) * 100, 2),
        "shap_breakdown": shap_breakdown,
        "recommendations": recommendations,
    }


@app.get("/model-comparison")
async def model_comparison(force: bool = False):
    """
    Returns validation metrics for all registered models on CVD_Dataset.csv.
    """
    return _compute_model_comparison(force=force)

@app.get("/health")
def health():
    return{
        'Status': 'OK',
        'Version': '1.0.0'
    }
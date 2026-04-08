import streamlit as st
import pandas as pd
import requests

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="CVD Risk Predictor", layout="wide")

API_URL = "http://127.0.0.1:8003/predict"

# Available models (you can expand later)
MODEL_OPTIONS = {
    "Tuned Ensemble (Best)": "tuned_ensemble2.pkl",
    "Random Forest": "tuned_rf.pkl",
    "XGBoost": "tuned_xgb.pkl",
    "Base Ensemble": "tuned_ensemble.pkl",
    "LightGBM": "tuned_lgbm.pkl",
    "Decision Tree": "tuned_dt.pkl"
}

# ==============================
# TITLE
# ==============================
st.markdown(
    "<h1 style='text-align: center;'>💓 Cardiovascular Disease Risk Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>AI-powered early risk detection system</h4>",
    unsafe_allow_html=True
)
st.markdown(
    "<h5 style='text-align: center;'>============================NOTE: Please enter all details in order to get best prediction.============================</h5>",
    unsafe_allow_html=True
)

# ==============================
# MODEL SELECTION
# ==============================
st.sidebar.header("⚙️ Settings")

selected_model_name = st.sidebar.selectbox(
    "Select Prediction Model",
    list(MODEL_OPTIONS.keys())
)

selected_model_file = MODEL_OPTIONS[selected_model_name]

st.sidebar.info(f"Using: {selected_model_name}")

st.sidebar.divider()
show_model_comparison = st.sidebar.checkbox("Compare all models (validation)", value=False)

# ==============================
# INPUT FORM
# ==============================
st.header("🧾 Enter Patient Details")

col1, col2 = st.columns(2)

# -------- Column 1 --------
with col1:
    st.subheader("Demographics")
    age = st.number_input("Age", 18, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
    urban = st.selectbox("Area", ["Urban", "Rural"])
    socio = st.selectbox("Socioeconomic Status", ["Lower", "Lower-Middle", "Middle", "Upper-Middle", "Upper"])
    occupation = st.selectbox("Occupation", ["Unemployed", "Small_Business", "Labor_Worker", "IT_Professional", "Students"])

# -------- Column 2 --------
with col2:
    st.subheader("Lifestyle")
    diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Eggetarian"])
    activity = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
    steps = st.number_input("Daily Steps", 0, 50000, 5000)
    life_row1_col1, life_row1_col2 = st.columns(2)
    with life_row1_col1:
        junk = st.slider("Junk Food Frequency", 0, 10, 5)
    with life_row1_col2:
        screen = st.slider("Screen Time (hrs)", 0.0, 24.0, 5.0)
    sleep = st.slider("Sleep Hours", 0.0, 24.0, 7.0)
    stress = st.slider("Stress Level", 1, 10, 5)
    # Visual balance: align bottom edge with Demographics column.
    st.markdown("<div style='height: 2.2rem;'></div>", unsafe_allow_html=True)

# -------- Full-width row below --------
st.subheader("Medical + Biomarkers")
med1, med2 = st.columns(2)

with med1:
    smoking = st.selectbox("Smoking", ["Never", "Occasional", "Regular"])
    alcohol = st.selectbox("Alcohol", ["Low", "Moderate", "High"])
    family = st.selectbox("Family History CVD", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    pcos = st.selectbox("PCOS", [0, 1])
    pollution = st.selectbox("Pollution Exposure", ["Low", "Moderate", "High"])
    work = st.selectbox("Work Stress Type", ["Sedentary", "Physical", "Shift-based"])
    trig = st.number_input("Triglycerides", 20.0, 500.0, 150.0)

with med2:
    bmi = st.number_input("BMI", 10.0, 60.0, 24.5)
    sys_bp = st.number_input("Systolic BP", 80, 200, 120)
    dia_bp = st.number_input("Diastolic BP", 40, 130, 80)
    hr = st.number_input("Heart Rate", 40, 150, 70)
    sugar = st.number_input("Fasting Sugar", 50.0, 300.0, 100.0)
    hba1c = st.number_input("HbA1c", 3.0, 15.0, 5.5)
    chol = st.number_input("Total Cholesterol", 50.0, 400.0, 200.0)
    ldl = st.number_input("LDL", 20.0, 300.0, 130.0)
    hdl = st.number_input("HDL", 10.0, 150.0, 50.0)

# ==============================
# PREDICT BUTTON
# ==============================
st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
if st.button("Predict Risk"):

    payload = {
        "Age": age,
        "Gender": gender,
        "Region": region,
        "Urban_Rural": urban,
        "Socioeconomic_Status": socio,
        "Occupation": occupation,
        "Diet_Type": diet,
        "Junk_Food_Frequency": junk,
        "Physical_Activity_Level": activity,
        "Daily_Steps": steps,
        "Screen_Time_Hours": screen,
        "Sleep_Hours": sleep,
        "Stress_Level": stress,
        "Smoking_Status": smoking,
        "Alcohol_Consumption": alcohol,
        "Family_History_CVD": family,
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "PCOS": pcos,
        "BMI": bmi,
        "Blood_Pressure_Systolic": sys_bp,
        "Blood_Pressure_Diastolic": dia_bp,
        "Resting_Heart_Rate": hr,
        "Fasting_Blood_Sugar": sugar,
        "HbA1c": hba1c,
        "Total_Cholesterol": chol,
        "LDL": ldl,
        "HDL": hdl,
        "Triglycerides": trig,
        "Air_Pollution_Exposure": pollution,
        "Work_Stress_Type": work,
        "model_name": selected_model_file
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction Successful!")

            col1, col2, col3 = st.columns(3)

            cvd_risk = result.get("CVD_Risk", None)
            risk_pct = result.get("risk_probability", None)
            risk_text = result.get("risk_probability_text", None)

            col1.metric("Risk probability", risk_text if risk_text is not None else (f"{risk_pct}%" if risk_pct is not None else "—"))
            col2.metric("Predicted class (0/1)", "—" if cvd_risk is None else str(cvd_risk))
            col3.metric("Category", result["risk_level"])

            # Explanation (user-friendly)
            breakdown = result.get("shap_breakdown", None)
            shap_data = result.get("shap_values", None)

            if breakdown is not None and len(breakdown) > 0:
                st.subheader("🔍 Why this prediction? (Easy explanation)")

                base_pct = result.get("shap_base_probability", None)
                if base_pct is not None:
                    st.caption(f"Baseline risk: **{base_pct:.2f}%** → Your risk: **{risk_pct:.2f}%**")

                df_breakdown = pd.DataFrame(breakdown)
                df_breakdown["pct_points"] = pd.to_numeric(df_breakdown["pct_points"], errors="coerce").fillna(0.0)
                # Keep very small effects; we'll render with higher precision if needed.
                df_breakdown = df_breakdown[df_breakdown["pct_points"].abs() > 1e-10]

                df_up = (
                    df_breakdown[df_breakdown["pct_points"] > 0]
                    .sort_values("pct_points", ascending=False)
                    .reset_index(drop=True)
                )
                df_down = (
                    df_breakdown[df_breakdown["pct_points"] < 0]
                    .assign(pct_points=lambda d: d["pct_points"].abs())
                    .sort_values("pct_points", ascending=False)
                    .reset_index(drop=True)
                )

                def _fmt_pp(x: float) -> str:
                    ax = abs(float(x))
                    # For tiny contributions, show more decimals so it doesn't look like "0.00"
                    return f"{ax:.4f}%" if ax < 0.05 else f"{ax:.2f}%"

                if not df_up.empty:
                    st.markdown("**What increased your risk**")
                    lines = []
                    for i, row in enumerate(df_up.itertuples(index=False), start=1):
                        lines.append(f"{i}. **+{_fmt_pp(row.pct_points)}** because of **{row.feature}**")
                    st.markdown("\n".join(lines))

                if not df_down.empty:
                    st.markdown("**What decreased your risk**")
                    lines = []
                    for i, row in enumerate(df_down.itertuples(index=False), start=1):
                        lines.append(f"{i}. **-{_fmt_pp(row.pct_points)}** because of **{row.feature}**")
                    st.markdown("\n".join(lines))

                # Simple chart: show absolute magnitude, with direction in a separate column.
                df_chart = df_breakdown.copy()
                df_chart["direction"] = df_chart["pct_points"].apply(lambda x: "Increase" if x > 0 else "Decrease")
                df_chart["magnitude"] = df_chart["pct_points"].abs()
                df_chart = df_chart.sort_values("magnitude", ascending=False).head(10)
                st.bar_chart(df_chart.set_index("feature")[["magnitude"]])

                with st.expander("See detailed breakdown (percentage points)"):
                    df_breakdown_view = (
                        df_breakdown.rename(columns={"feature": "Feature", "pct_points": "Δ Risk (pp)"})
                        .sort_values("Δ Risk (pp)", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.dataframe(
                        df_breakdown_view.style.format({"Δ Risk (pp)": "{:+.6f}"})
                    )

                # ==============================
                # PERSONALIZED RECOMMENDATIONS
                # ==============================
                recommendations = result.get("recommendations", [])
                if recommendations:
                    st.subheader("Personalized Recommendations (WHO Guidelines)")
                    st.caption("Order matches the factors listed in 'What increased your risk'.")

                    for i, rec in enumerate(recommendations, start=1):
                        title = rec.get("feature", "Unknown")
                        priority = rec.get("priority", "Medium")
                        impact = float(rec.get("impact", 0.0) or 0.0)
                        with st.expander(f"{i}. {title} ({priority} priority, +{impact:.2f} pp)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Current:** {rec.get('current_value', 'N/A')}")
                                st.markdown(f"**Target:** {rec.get('target', 'N/A')}")
                            with col2:
                                st.markdown(f"**Action:** {rec.get('action', 'No action specified')}")
                                safety = rec.get("safety_flag")
                                if safety:
                                    st.error(f"**Safety Alert:** {safety}")
                else:
                    st.info("No specific recommendations available for this profile. Your current values appear to be within healthy ranges.")

            else:
                # If backend sends an explanation error, show a friendly message.
                if isinstance(shap_data, dict) and "error" in shap_data:
                    st.warning(f"Explanation not available: {shap_data['error']}")
                else:
                    st.info("Explanation is unavailable (effects may be extremely small for this input). Try changing a few inputs and predict again.")

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Connection Error: {e}")

# ==============================
# MODEL COMPARISON (VALIDATION)
# ==============================
if show_model_comparison:
    st.markdown(
    "<h3 style='text-align: center;'>Model comparison (validation on dataset)</h3>",
    unsafe_allow_html=True
    )
    st.markdown(
    "<p style='text-align: center; font-size: 14px; color: gray;'>"
    "Compares all 6 models using validation metrics from `CVD_Dataset.csv` via the backend."
    "</p>",
    unsafe_allow_html=True
    )
    if st.button("Run Comparison"):
        try:
            r = requests.get("http://127.0.0.1:8003/model-comparison", params={"force": "true"}, timeout=120)
            if r.status_code != 200:
                st.error(f"Backend error: {r.text}")
            else:
                st.session_state["model_comparison"] = r.json()
        except Exception as e:
            st.error(f"Connection Error: {e}")

    data = st.session_state.get("model_comparison")
    if data and "models" in data:
        df_models = pd.DataFrame(data["models"])
        df_ok = df_models[df_models.get("available", False) == True].copy()
        df_bad = df_models[df_models.get("available", False) != True].copy()

        if not df_ok.empty:
            cols = ["model_key", "roc_auc", "accuracy", "f1", "precision", "recall"]
            for c in cols:
                if c not in df_ok.columns:
                    df_ok[c] = None
            df_ok = df_ok[cols].sort_values(["roc_auc", "accuracy"], ascending=False).reset_index(drop=True)

            st.subheader("Results (models with higher ROC-AUC & Recall are better)")
            st.dataframe(df_ok.style.format({"roc_auc": "{:.4f}", "accuracy": "{:.4f}", "f1": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}"}))

        if not df_bad.empty:
            st.subheader("Unavailable / failed models")
            cols = ["model_key", "error"]
            for c in cols:
                if c not in df_bad.columns:
                    df_bad[c] = ""
            st.dataframe(df_bad[cols])

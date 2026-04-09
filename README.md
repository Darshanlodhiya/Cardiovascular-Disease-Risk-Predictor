# Cardiovascular Disease (CVD) Risk Predictor

End-to-end ML application for CVD risk estimation with:

- **FastAPI backend** for prediction, SHAP explainability, and recommendation generation
- **Streamlit frontend** for interactive data entry and result visualization
- **Notebook-based training pipeline** (`Model_From_New_Dataset.ipynb`) documenting model development

---

## 1) Project Purpose

This project predicts whether a patient profile is at higher CVD risk and explains the prediction in understandable terms.

The app emphasizes:

- Probability-based risk output (not only class labels)
- SHAP-driven explanation of risk-increasing/decreasing factors
- Patient-centric, practical recommendations for **modifiable** factors that increased risk

> Clinical note: This tool supports screening and education, not diagnosis. Medical decisions should be confirmed by a qualified clinician.

---

## 2) Repository Structure

- `main.py` - FastAPI backend (model loading, preprocessing, prediction, SHAP, recommendations, model comparison)
- `app.py` - Streamlit frontend UI
- `Model_From_New_Dataset.ipynb` - model training + evaluation workflow
- `CVD_Dataset.csv` - training/evaluation dataset
- `tuned_ensemble2.pkl` - default prediction model used by backend
- `tuned_rf.pkl`, `tuned_xgb.pkl`, `tuned_lgbm.pkl`, `tuned_dt.pkl`, `tuned_ensemble.pkl` - alternative trained models
- `requirements.txt` - Python dependencies
- `debug_call.py` - helper script for backend call testing

---

## 3) High-Level Architecture

1. User enters profile in Streamlit (`app.py`).
2. Frontend sends JSON payload to `POST /predict`.
3. Backend preprocesses data to training-compatible feature format.
4. Selected model predicts class + risk probability.
5. Backend computes SHAP attributions and a probability-point breakdown.
6. Backend generates personalized recommendations from positive SHAP contributors.
7. Frontend renders risk cards, explanation, charts, and recommendations.

---

## 4) Data & Feature Engineering (from notebook)

The training notebook (`Model_From_New_Dataset.ipynb`) shows:

### 4.1 Core pipeline stages

- Data loading from `CVD_Dataset.csv`
- EDA and visualization
- Feature engineering
- Train/validation/test split
- Class imbalance handling via class weights
- Outlier handling (IQR-based capping)
- Categorical encoding
- Hyperparameter tuning
- Ensemble modeling
- Threshold tuning and evaluation

### 4.2 Engineered features used by app/backend

The backend mirrors notebook feature logic, including:

- `Metabolic Syndrome` (derived binary feature)
- `Cardio_Risk_Score` (composite risk count)
- `Pulse_Pressure` (`Systolic - Diastolic`)
- `Sleep_Risk` (sleep <6h or >9h)
- `High_Screen_Time` (>8h)

### 4.3 Data split strategy

Notebook split (stratified):

- Test split: `15%`
- Validation split: `15%` of remaining main data
- Final effective train/valid/test ratio is approximately `72.25 / 12.75 / 15`

---

## 5) Model Development Summary (from notebook)

Models explored/tuned include:

- Random Forest
- XGBoost
- LightGBM
- Decision Tree
- Soft Voting Ensemble

The notebook includes:

- Hyperparameter search with stratified CV
- Precision-Recall based threshold tuning (instead of fixed `0.5`)
- ROC/AUC, F1, confusion-matrix-based evaluation
- Saved tuned model artifacts via `joblib`

In deployment (`main.py`), default is:

- `tuned_ensemble2.pkl` (fallback to next available model if unavailable)

---

## 6) Backend (FastAPI)

### 6.1 Main endpoints

- `POST /predict`
  - Input: patient profile matching `UserInput` schema in `main.py`
  - Output includes:
    - `CVD_Risk` (0/1)
    - `risk_probability` and `risk_level`
    - `shap_values`
    - `shap_base_probability`
    - `shap_breakdown`
    - `recommendations`

- `GET /model-comparison`
  - Evaluates all registered models on `CVD_Dataset.csv`
  - Returns validation metrics and confusion matrix info

### 6.2 Explainability logic

Backend SHAP strategy:

- First attempts `TreeExplainer` (fast when model type supports it)
- Falls back to permutation-based model-agnostic SHAP when needed
- Converts SHAP effects into signed **probability percentage points** for easier interpretation

### 6.3 Recommendation engine behavior

Recommendations are generated from `shap_breakdown` by:

- Selecting positive contributors (`pct_points > 0`)
- Filtering to modifiable factors
- Preserving the same order as risk-increase explanation
- Returning practical actions + personalized targets + safety flags

Safety alert examples include severe BP/glucose/lipid ranges.

---

## 7) Frontend (Streamlit)

The UI provides:

- Patient input form (demographics, lifestyle, medical/biomarkers)
- Risk probability + category cards
- SHAP explanation:
  - “What increased risk”
  - “What decreased risk”
  - bar chart and detailed table
- Personalized recommendation cards with:
  - current value
  - target
  - action
  - priority/impact
  - safety alerts
- Optional model comparison view

---

## 8) Run Locally

### 8.1 Prerequisites

- Python 3.11+ recommended
- `pip`

### 8.2 Install

```bash
python -m pip install -r requirements.txt
```

### 8.3 Start backend

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8003
```

### 8.4 Start frontend

```bash
python -m streamlit run app.py --server.port 8501
```

Open: [http://localhost:8501](http://localhost:8501)

---

## 8.5) Run With Docker (Containerized)

This project provides a multi-service Docker setup:

- `backend` (FastAPI) on port `8003`
- `frontend` (Streamlit) on port `8501`

### Build and start

```bash
docker compose up --build
```

### Run in detached mode

```bash
docker compose up --build -d
```

### Stop containers

```bash
docker compose down
```

### View container logs

```bash
docker compose logs -f
```

Open: [http://localhost:8501](http://localhost:8501)

---

## 9) Model Files and Selection

Available model artifacts registered in backend:

- `tuned_ensemble2.pkl` (default)
- `tuned_rf.pkl`
- `tuned_xgb.pkl`
- `tuned_ensemble.pkl`
- `tuned_lgbm.pkl`
- `tuned_dt.pkl`

Frontend lets users pick model; selected file is sent as `model_name`.

---

## 10) Important Implementation Notes

- Backend includes compatibility handling for serialized wrappers that reference `__main__.ThresholdModel`.
- If model loading fails due to missing optional dependencies, API remains bootable and falls back where possible.
- SHAP for unsupported model classes (e.g., some ensembles) uses fallback explainers.
- Some environments may show sklearn unpickle version warnings if runtime sklearn differs from training version.

---

## 11) Troubleshooting

### Backend not starting

- Ensure required model `.pkl` files are present in project root.
- Reinstall dependencies:
  ```bash
  python -m pip install -r requirements.txt
  ```

### SHAP unavailable for some model

- This can happen for unsupported estimator types.
- Backend fallback explainer is used automatically.

### Empty/limited recommendations

- Recommendations only come from **positive SHAP** contributors and modifiable features.
- If profile has few positive modifiable contributors, list may be short.

### Model comparison is slow

- Endpoint computes metrics across multiple models and dataset rows.
- Use cached result unless forcing refresh.

---

## 12) Development Tips

- Quick syntax check:
  ```bash
  python -m py_compile main.py app.py
  ```
- Test backend endpoint manually:
  ```bash
  python debug_call.py
  ```

---

## 13) Future Improvements (Suggested)

- Add structured API response models with Pydantic for output validation
- Add pytest coverage for preprocessing/recommendation ordering
- Persist model comparison cache across process restarts
- Add user authentication and longitudinal patient history tracking
- Export reports (PDF/JSON) for clinician workflows

---

## 14) Disclaimer

This software is for educational and risk-screening assistance. It is **not** a substitute for professional diagnosis, treatment, or emergency care.


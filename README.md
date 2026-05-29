# Cardiovascular Disease (CVD) Risk Predictor

Live Link: http://51.21.161.187:8501/

An end-to-end machine learning application that estimates cardiovascular disease risk, explains *why* the model predicted that risk, and suggests practical next actions.

This project combines:

- **FastAPI backend** for prediction, explainability, and recommendation logic
- **Streamlit frontend** for an easy-to-use clinical-style interface
- **Model training workflow** in `Model_From_New_Dataset.ipynb`

> This tool is for risk screening and education support. It is not a medical diagnosis system.

---

## 1) What this project does

For each patient profile, the app provides:

1. **Predicted risk class** (`0` or `1`)
2. **Predicted risk probability** (for example `37.76%`)
3. **Risk category** (`Low`, `Moderate`, `High`, `Very High`)
4. **Interpretability output**:
   - What increased risk
   - What decreased risk
   - How strongly each factor influenced this prediction
5. **Personalized recommendations** focused on modifiable factors

The main design goal is to keep predictions understandable, actionable, and safe to interpret.

---

## 2) Project structure

- `app.py`: Streamlit frontend
- `main.py`: FastAPI backend, preprocessing, model loading, SHAP logic, recommendations
- `requirements.txt`: Python dependencies
- `docker-compose.yml`: multi-service container orchestration
- `Dockerfile`: container build instructions
- `.dockerignore`: excludes unnecessary files from Docker build context
- `debug_call.py`: helper script for backend test calls
- `Model_From_New_Dataset.ipynb`: training and model development notebook
- `CVD_Dataset.csv`: dataset used for training/evaluation and model comparison endpoint
- `tuned_ensemble2.pkl`, `tuned_rf.pkl`, `tuned_xgb.pkl`, `tuned_ensemble.pkl`, `tuned_lgbm.pkl`, `tuned_dt.pkl`: trained model files

---

## 3) End-to-end workflow (how everything works)

### Step 1: User input

The user enters demographics, lifestyle, medical history, and biomarker values in the Streamlit form.

### Step 2: Frontend request

`app.py` sends JSON payload to backend endpoint:

- `POST /predict`

### Step 3: Backend preprocessing

`main.py`:

- Encodes categorical features (label + one-hot)
- Builds engineered features such as:
  - `Metabolic Syndrome`
  - `Cardio_Risk_Score`
  - `Pulse_Pressure`
  - `Sleep_Risk`
  - `High_Screen_Time`
- Aligns feature columns with selected model training schema

### Step 4: Prediction

Backend selects model (`tuned_ensemble2.pkl` by default, with fallback behavior), then computes:

- predicted class
- predicted probability
- risk category

### Step 5: Explainability

Backend computes SHAP values:

- tries `TreeExplainer` first
- falls back to model-agnostic permutation SHAP if needed

Then it converts model-space SHAP effects into **probability percentage points** for easier interpretation.

### Step 6: Recommendations

Backend creates recommendation cards for factors that:

- increased the risk
- are modifiable
- have meaningful contribution

### Step 7: Frontend display

Streamlit renders:

- risk cards
- increase/decrease explanation lists
- bar chart + table
- recommendations with targets/actions/safety alerts

---

## 4) How to read the prediction (beginner-friendly)

This section is based on your updated interpretation guide.

### Simple mental model

- **Your risk (example: 37.76%)** = model estimate for this specific profile
- **Baseline risk (example: 62.59%)** = model starting point for a typical profile in training data (not personal baseline)
- The two lists explain movement from baseline:
  - **What increased your risk** -> pushes prediction up
  - **What decreased your risk** -> pushes prediction down

### What users can safely conclude

1. Largest positive values are strongest reasons for higher predicted risk
2. Modifiable factors deserve highest action priority (LDL, BP, sugar, BMI, activity, sleep, alcohol)
3. Largest negative values show protective factors in this profile
4. Bigger absolute percentage-point contribution means stronger influence on this prediction

### What users should not conclude

1. Baseline is not personal default risk
2. Feature contribution is not proof of biological causation
3. Output is decision support, not diagnosis

### One-line interpretation template

`My estimated risk is X%. The model suggests risk is increased mainly by A/B/C and reduced by D/E/F.`

---

## 5) Frontend details (`app.py`)

UI capabilities:

- Full patient input form (demographic + lifestyle + medical + biomarkers)
- Model selector in sidebar
- Predict button for single-case risk inference
- Explainability breakdown:
  - increasing factors
  - decreasing factors
  - top contribution chart
  - detailed percentage-point table
- Personalized recommendation cards
- Optional cross-model comparison view

Backend URL is configurable via environment variable:

- `API_BASE_URL` (default: `http://127.0.0.1:8003`)

---

## 6) Backend details (`main.py`)

### Main APIs

- `POST /predict`
  - input: patient profile schema (`UserInput`)
  - output:
    - `CVD_Risk`
    - `risk_probability`
    - `risk_probability_text`
    - `risk_level`
    - `shap_values`
    - `shap_base_probability`
    - `shap_breakdown`
    - `recommendations`

- `GET /model-comparison`
  - computes evaluation metrics for all registered models on `CVD_Dataset.csv`

- `GET /health`
  - health endpoint for monitoring and Docker healthchecks

### Explainability and recommendation behavior

- SHAP is calculated per prediction
- Contributions are converted to signed probability percentage points
- Recommendations follow risk-increasing factors order
- Safety flags are attached for severe values (for example very high BP, LDL, glucose, HbA1c)

---

## 7) Run locally (without Docker)

### Prerequisites

- Python 3.11+
- `pip`

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Start backend

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8003
```

### Start frontend

```bash
python -m streamlit run app.py --server.port 8501
```

Open:

- Frontend: [http://localhost:8501](http://localhost:8501)
- Backend docs: [http://127.0.0.1:8003/docs](http://127.0.0.1:8003/docs)

---

## 8) Run with Docker (full containerized setup)

This project is fully containerized with two services:

- `backend` on port `8003`
- `frontend` on port `8501`

### Build and run

```bash
docker compose up --build
```

### Run in detached mode

```bash
docker compose up --build -d
```

### Use Docker Hub images (no local build)

Set image names, then pull and run:

```bash
# PowerShell
$env:BACKEND_IMAGE="YOUR_DOCKERHUB_USERNAME/cvd-project-backend:latest"
$env:FRONTEND_IMAGE="YOUR_DOCKERHUB_USERNAME/cvd-project-frontend:latest"
docker compose pull
docker compose up -d
```

For Bash:

```bash
export BACKEND_IMAGE="YOUR_DOCKERHUB_USERNAME/cvd-project-backend:latest"
export FRONTEND_IMAGE="YOUR_DOCKERHUB_USERNAME/cvd-project-frontend:latest"
docker compose pull
docker compose up -d
```

### Check status

```bash
docker compose ps
```

### View logs

```bash
docker compose logs -f
```

### Stop everything

```bash
docker compose down
```

Open:

- Frontend: [http://localhost:8501](http://localhost:8501)
- Backend: [http://localhost:8003](http://localhost:8003)

---

## 9) Model files and selection

Configured model artifacts:

- `tuned_ensemble2.pkl` (default)
- `tuned_rf.pkl`
- `tuned_xgb.pkl`
- `tuned_ensemble.pkl`
- `tuned_lgbm.pkl`
- `tuned_dt.pkl`

Frontend sends selected model file name as `model_name` to backend.

If some model cannot load, backend keeps running and falls back to available models when possible.

---

## 10) Troubleshooting

### Backend fails to start

- Confirm model `.pkl` files exist in project root
- Confirm `CVD_Dataset.csv` exists (needed for model comparison and some explainability paths)
- Reinstall dependencies:

```bash
python -m pip install -r requirements.txt
```

### Docker services are unhealthy

- Check status:

```bash
docker compose ps
```

- Inspect logs:

```bash
docker compose logs backend --tail 200
docker compose logs frontend --tail 200
```

### SHAP explanation missing/limited

- Certain model types can require fallback explainer paths
- Very small feature effects can appear as near-zero contributions

### sklearn version warning while loading pickles

- Warning indicates runtime/version mismatch with training environment
- Usually still runs, but best practice is to align versions for strict reproducibility

---

## 11) Development helpers

### Syntax check

```bash
python -m py_compile main.py app.py
```

### Quick backend call test

```bash
python debug_call.py
```

---

## 12) Future improvements

- Add stricter API response models for output validation
- Add unit tests for preprocessing and recommendation ordering
- Persist model-comparison cache across restarts
- Add authentication and patient history tracking
- Add exportable clinical summary report (JSON/PDF)

---

## 13) Clinical and safety disclaimer

This software supports education and risk screening only. It is **not** a substitute for professional medical diagnosis, treatment planning, or emergency care.


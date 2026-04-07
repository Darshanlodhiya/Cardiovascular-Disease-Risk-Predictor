# Cardiovascular Disease Risk Predictor

Streamlit UI + FastAPI backend for predicting cardiovascular disease (CVD) risk and showing a user-friendly SHAP explanation.

## Run locally

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Start the backend (FastAPI)

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

### 3) Start the frontend (Streamlit)

```bash
python -m streamlit run app.py --server.port 8502
```

Then open `http://localhost:8502`.

## Notes

- Model files (`*.pkl`) are ignored by default in `.gitignore` because they can be large.  
  If you want them on GitHub, use **Git LFS** or upload them as **GitHub Release assets**.


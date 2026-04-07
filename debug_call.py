import json
import sys

import requests


def main() -> int:
    payload = {
        "Age": 25,
        "Gender": "Male",
        "Region": "North",
        "Urban_Rural": "Urban",
        "Socioeconomic_Status": "Middle",
        "Occupation": "IT_Professional",
        "Diet_Type": "Vegetarian",
        "Junk_Food_Frequency": 5,
        "Physical_Activity_Level": "Low",
        "Daily_Steps": 5000,
        "Screen_Time_Hours": 5.0,
        "Sleep_Hours": 7.0,
        "Stress_Level": 5,
        "Smoking_Status": "Never",
        "Alcohol_Consumption": "Low",
        "Family_History_CVD": 0,
        "Diabetes": 0,
        "Hypertension": 0,
        "PCOS": 0,
        "BMI": 24.5,
        "Blood_Pressure_Systolic": 120,
        "Blood_Pressure_Diastolic": 80,
        "Resting_Heart_Rate": 70,
        "Fasting_Blood_Sugar": 100.0,
        "HbA1c": 5.5,
        "Total_Cholesterol": 200.0,
        "LDL": 130.0,
        "HDL": 50.0,
        "Triglycerides": 150.0,
        "Air_Pollution_Exposure": "Low",
        "Work_Stress_Type": "Sedentary",
        "model_name": "tuned_ensemble2.pkl",
    }

    for port in (8000, 8001, 8002):
        url = f"http://127.0.0.1:{port}/predict"
        try:
            r = requests.post(url, json=payload, timeout=10)
            print(f"\n=== {url} ===")
            print("status:", r.status_code)
            try:
                print(json.dumps(r.json(), indent=2)[:5000])
            except Exception:
                print(r.text[:5000])
        except Exception as e:
            print(f"\n=== {url} ===")
            print("ERROR:", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

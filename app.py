from typing import Any
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException


MODEL_NAME = "BankChurnLab4Model"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "bank_churn_lab4_model.pkl"

REQUIRED_FEATURES = [
    "credit_score",
    "geography",
    "gender",
    "age",
    "tenure_years",
    "balance",
    "num_products",
    "has_cr_card",
    "is_active_member",
    "estimated_salary",
    "monthly_transactions",
    "complaints_last_year",
]

model = None
model_loaded = False

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True

    print("===================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("Model:", MODEL_NAME)
    print("Path:", MODEL_PATH)
    print("===================================")

except Exception as e:
    print("MODEL LOADING FAILED:", e)


app = FastAPI(
    title="Bank Churn Prediction API",
    description="Bank Customer Churn Prediction using the LAB4 trained model",
    version="1.0",
)


@app.get("/")
def home():
    return {
        "message": "Bank Churn Prediction API is running",
        "model": MODEL_NAME,
        "model_loaded": model_loaded,
    }


@app.get("/health")
def health():
    if model_loaded:
        return {
            "status": "healthy",
            "model_loaded": True,
            "model": MODEL_NAME,
        }

    raise HTTPException(
        status_code=503,
        detail="Model is not loaded",
    )


@app.get("/features")
def features():
    return {
        "required_features": REQUIRED_FEATURES,
    }


@app.post("/predict")
def predict(data: dict[str, Any]):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    missing = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in data
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features",
                "missing_features": missing,
            },
        )

    try:
        input_df = pd.DataFrame(
            [[data[feature] for feature in REQUIRED_FEATURES]],
            columns=REQUIRED_FEATURES,
        )

        prediction = model.predict(input_df)[0]

        result = {
            "prediction": int(prediction),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            result["probability_no_churn"] = float(
                probabilities[classes.index(0)]
            )
            result["probability_churn"] = float(
                probabilities[classes.index(1)]
            )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
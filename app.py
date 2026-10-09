
from typing import Any

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException

# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------

mlflow.set_tracking_uri(
    "sqlite:///D:/kesava 2/231FA04B56/mlflow.db"
)

MODEL_NAME = "BankChurnLab4Model"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

# --------------------------------------------------
# Load model
# --------------------------------------------------

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    model_loaded = True

    print("===================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("Model:", MODEL_NAME)
    print("URI:", MODEL_URI)
    print("===================================")

except Exception as e:
    model = None
    model_loaded = False
    print("MODEL LOADING FAILED:", e)

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Bank Churn Prediction API",
    description="Bank Customer Churn Prediction using MLflow",
    version="1.0"
)

# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Bank Churn Prediction API is running",
        "model": MODEL_NAME,
        "alias": "champion"
    }

# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():

    if model_loaded:
        return {
            "status": "healthy",
            "model_loaded": True,
            "model": MODEL_NAME,
            "model_uri": MODEL_URI
        }

    raise HTTPException(
        status_code=503,
        detail="Model is not loaded"
    )

# --------------------------------------------------
# Features
# --------------------------------------------------

@app.get("/features")
def features():

    return {
        "required_features": [
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
            "complaints_last_year"
        ]
    }

# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
def predict(data: dict[str, Any]):

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    required_features = [
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
        "complaints_last_year"
    ]

    missing = [
        feature
        for feature in required_features
        if feature not in data
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features",
                "missing_features": missing
            }
        )

    try:

        input_df = pd.DataFrame([data])

        input_df = input_df[required_features]

        prediction = model.predict(input_df)[0]

        result = {
            "prediction": int(prediction)
        }

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_df)[0]

            result["probability_no_churn"] = float(probability[0])
            result["probability_churn"] = float(probability[1])

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

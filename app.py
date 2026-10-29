
from typing import Any
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("bank-churn-api")


MODEL_NAME = "BankChurnLab4Model"
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "bank_churn_lab4_model.pkl"
)

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
    logger.info(
        "event=model_loaded model=%s model_path=%s",
        MODEL_NAME,
        MODEL_PATH.name,
    )
except Exception:
    logger.exception(
        "event=model_load_failed model=%s model_path=%s",
        MODEL_NAME,
        MODEL_PATH.name,
    )


app = FastAPI(
    title="Bank Churn Prediction API",
    description="Bank Customer Churn Prediction using the LAB4 trained model",
    version="1.0",
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        log_level = (
            logging.WARNING
            if response.status_code >= 400
            else logging.INFO
        )

        logger.log(
            log_level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    logger.warning(
        "event=request_validation_failed method=%s path=%s error_count=%s",
        request.method,
        request.url.path,
        len(exc.errors()),
    )

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
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
        logger.info("event=health_check status=healthy")
        return {
            "status": "healthy",
            "model_loaded": True,
            "model": MODEL_NAME,
        }

    logger.error("event=health_check status=unhealthy model_loaded=false")

    raise HTTPException(
        status_code=503,
        detail="Model is not loaded",
    )


@app.get("/features")
def features():
    logger.info("event=features_requested")
    return {
        "required_features": REQUIRED_FEATURES,
    }


@app.post("/predict")
def predict(data: dict[str, Any]):
    if not model_loaded:
        logger.error("event=prediction_failed reason=model_not_loaded")
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
        logger.warning(
            "event=prediction_validation_failed reason=missing_features missing_count=%s",
            len(missing),
        )

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

        logger.info("event=prediction_succeeded")

        return result

    except HTTPException:
        raise

    except Exception:
        logger.exception("event=prediction_failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error",
        )


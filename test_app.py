from fastapi.testclient import TestClient

from app import app, model_loaded

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Bank Churn Prediction API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_features():
    response = client.get("/features")

    assert response.status_code == 200

    features = response.json()["required_features"]

    assert len(features) == 12
    assert "credit_score" in features
    assert "complaints_last_year" in features


def test_predict():
    payload = {
        "credit_score": 450,
        "geography": "Germany",
        "gender": "Male",
        "age": 60,
        "tenure_years": 2,
        "balance": 120000,
        "num_products": 1,
        "has_cr_card": 1,
        "is_active_member": 0,
        "estimated_salary": 50000,
        "monthly_transactions": 3,
        "complaints_last_year": 4,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability_no_churn"] <= 1
    assert 0 <= result["probability_churn"] <= 1


def test_predict_missing_features():
    payload = {
        "credit_score": 650,
        "geography": "France",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400

    result = response.json()

    assert result["detail"]["message"] == "Missing required features"


def test_model_loaded():
    assert model_loaded is True
from fastapi.testclient import TestClient
import pytest
from sentiment_app.app import app

def test_predict_endpoint_success():
    """Test sprawdza, czy aplikacja poprawnie odpowiada na poprawne zapytanie."""
    payload = {"text": "This MLOps course is absolutely amazing and helpful!"}
    
    # Użycie bloku 'with' zmusza FastAPI do wykonania zdarzeń startup (wczytanie modeli ONNX)
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        # Sprawdzamy status HTTP 200 (OK)
        assert response.status_code == 200
        
        # Sprawdzamy strukturę odpowiedzi JSON
        data = response.json()
        assert "text" in data
        assert "sentiment" in data
        assert data["text"] == payload["text"]
        assert data["sentiment"] in ["positive", "neutral", "negative"]

def test_predict_endpoint_empty_text():
    """Test sprawdza, czy aplikacja poprawnie odrzuca pusty tekst (status 400)."""
    payload = {"text": "   "}
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        # Sprawdzamy status 400
        assert response.status_code == 400
        assert "detail" in response.json()
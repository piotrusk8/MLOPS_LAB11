# Raport z wykonania zadania - Temat 11 (CD & Deployment)

**Kurs:** MLOps AGH
**Status wykonania:** Sukces (Zakończone)

### 1. Wynik wykonania potoku GitHub Actions
Kod źródłowy został pomyślnie oczyszczony z lokalnych sekretów i wypchnięty do nowego repozytorium. Potok CI/CD w GitHub Actions został skonfigurowany przy użyciu zaszyfrowanych zmiennych środowiskowych (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN). Wszystkie kroki testów jednostkowych oraz wdrożenia bezserwerowego na AWS Lambda kończą się sukcesem.

### 2. Wynik końcowy - Sukces inferencji modelu (Model Serving)
Aplikacja FastAPI poprawnie integruje modele ONNX (tokenizer.json, onnx_model.onnx, onnx_classifier.onnx) za pośrednictwem biblioteki onnxruntime.

Lokalne testy potoku inferencyjnego za pomocą 'uv run pytest' zakończyły się pełnym sukcesem (2 passed).

**Zapytanie testowe (cURL):**
``bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"This MLOps course is absolutely amazing and helpful!\"}"
``

**Zwrócona odpowiedź z serwera (Sukces inferencji):**
``json
{
  "text": "This MLOps course is absolutely amazing and helpful!",
  "sentiment": "neutral"
}
``

Aplikacja została poprawnie przygotowana do działania w architekturze Serverless (AWS Lambda) przy użyciu adaptera Mangum.
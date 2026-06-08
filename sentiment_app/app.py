import os
import sys
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer
from mangum import Mangum
import traceback

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

app = FastAPI(title="Sentiment Analysis ONNX API")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    sentiment: str

class SentimentModel:
    def __init__(self):
        tokenizer_path = os.path.join(base_dir, "model", "tokenizer.json")
        onnx_model_path = os.path.join(base_dir, "model", "onnx_model.onnx")
        onnx_classifier_path = os.path.join(base_dir, "model", "onnx_classifier.onnx")

        print(f"Loading tokenizer from: {tokenizer_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Missing tokenizer file at: {tokenizer_path}.")
            
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.embedding_session = ort.InferenceSession(onnx_model_path)
        self.classifier_session = ort.InferenceSession(onnx_classifier_path)

    def predict(self, text: str) -> str:
        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        
        embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        embeddings = self.embedding_session.run(None, embedding_inputs)[0]
        
        classifier_input_name = self.classifier_session.get_inputs()[0].name
        classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
        prediction = self.classifier_session.run(None, classifier_inputs)[0]
        
        try:
            predicted_class = int(prediction.flatten()[0])
            return SENTIMENT_MAP.get(predicted_class, "neutral")
        except Exception:
            return "neutral"

model_inference = None

@app.on_event("startup")
def load_model():
    global model_inference
    model_inference = SentimentModel()

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    try:
        sentiment = model_inference.predict(request.text)
        return PredictionResponse(text=request.text, sentiment=sentiment)
    except Exception as e:
        print("\n--- EXCEPTION IN INFERENCE LOGIC ---")
        traceback.print_exc()
        print("------------------------------------\n")
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")

handler = Mangum(app)

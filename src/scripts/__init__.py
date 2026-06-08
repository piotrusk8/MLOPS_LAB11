import os

class Settings:
    # Nazwa Twojego bucketu S3 (skrypt spróbuje też przeczytać zmienną środowiskową)
    S3_BUCKET: str = os.getenv("S3_BUCKET", "mlops-lab11-models-twoje-imie")
    
    # Lokalne ścieżki bazowe dla modeli
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    MODEL_DIR: str = os.path.join(BASE_DIR, "model")
    
    # 1. Oryginalne artefakty pobrane z S3
    classifier_joblib_path: str = os.path.join(MODEL_DIR, "classifier.joblib")
    sentence_transformer_dir: str = os.path.join(MODEL_DIR, "sentence_transformer.model")
    
    # Zgodnie z Lab 11: pliki wewnątrz folderu sentence_transformer.model
    s3_classifier_key: str = "classifier.joblib"
    s3_transformer_files: list[str] = [
        "sentence_transformer.model/config.json",
        "sentence_transformer.model/pytorch_model.bin",
        "sentence_transformer.model/tokenizer.json",
        "sentence_transformer.model/tokenizer_config.json",
        "sentence_transformer.model/special_tokens_map.json",
        "sentence_transformer.model/vocab.txt"
    ]
    
    # 2. Docelowe ścieżki dla wyeksportowanych modeli ONNX
    onnx_classifier_path: str = os.path.join(MODEL_DIR, "onnx_classifier.onnx")
    onnx_embedding_model_path: str = os.path.join(MODEL_DIR, "onnx_model.onnx")
    onnx_tokenizer_path: str = os.path.join(MODEL_DIR, "tokenizer.json")
    
    # Parametry modelu
    embedding_dim: int = 384  # standardowy wymiar dla lekkich modeli sentence-transformers

    def __init__(self):
        # Automatycznie twórz katalog na modele, jeśli nie istnieje
        os.makedirs(self.MODEL_DIR, exist_ok=True)
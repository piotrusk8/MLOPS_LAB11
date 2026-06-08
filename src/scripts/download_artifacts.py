import os
import boto3
from botocore.exceptions import ClientError
from settings import Settings

def download_file_from_s3(s3_client, bucket: str, s3_key: str, local_path: str):
    """Pobiera pojedynczy plik z S3 do wskazanej ścieżki lokalnej."""
    # Upewnij się, że katalog docelowy dla pliku istnieje
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    print(f"Pobieranie z S3: s3://{bucket}/{s3_key} -> {local_path}")
    try:
        s3_client.download_file(bucket, s3_key, local_path)
    except ClientError as e:
        print(f"Błąd podczas pobierania {s3_key}: {e}")
        raise e

def main():
    # Inicjalizacja konfiguracji ścieżek
    settings = Settings()
    
    # Utworzenie klienta boto3
    # Lokalnie pobierze on poświadczenia z Twojego pliku ~/.aws/credentials (skonfigurowanego przez aws configure)
    s3_client = boto3.client("s3")
    
    print(f"Rozpoczynam pobieranie artefaktów z bucketu: {settings.S3_BUCKET}")
    
    # 1. Pobieranie klasyfikatora (.joblib)
    download_file_from_s3(
        s3_client, 
        settings.S3_BUCKET, 
        settings.s3_classifier_key, 
        settings.classifier_joblib_path
    )
    
    # 2. Pobieranie wszystkich plików wchodzących w skład Sentence Transformera
    for s3_key in settings.s3_transformer_files:
        # Ustalamy lokalną ścieżkę: łączymy katalog główny z relatywną ścieżką pliku
        local_path = os.path.join(settings.BASE_DIR, s3_key)
        download_file_from_s3(s3_client, settings.S3_BUCKET, s3_key, local_path)
        
    print("Wszystkie artefakty zostały pomyślnie pobrane lokalnie!")

if __name__ == "__main__":
    main()
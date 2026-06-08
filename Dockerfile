# Etap budowania (Build stage)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Kopiowanie definicji zależności
COPY pyproject.toml uv.lock ./

# Synchronizacja wyłącznie grupy produkcyjnej (inference) bez instalowania samego projektu
RUN uv sync --frozen --group inference --no-install-project

# Etap uruchomieniowy (Runtime stage) dla AWS Lambda
FROM python:3.12-slim-bookworm

WORKDIR /app

# Kopiowanie wirtualnego środowiska z etapu builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Kopiowanie kodu aplikacji oraz wyeksponowanych artefaktów ONNX
COPY sentiment_app ./sentiment_app
COPY model ./model

# Konfiguracja punktu wejścia i komendy zgodnej z wymaganiami AWS Lambda kompatybilnej z Mangum
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["sentiment_app.app.handler"]
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    EMBEDDING_BACKEND=gemini \
    CHROMA_PERSIST_DIR=/tmp/chroma_data

WORKDIR /app

COPY requirements-render.txt ./
RUN pip install --no-cache-dir -r requirements-render.txt

COPY scripts ./scripts
COPY src ./src

CMD ["sh", "-c", "python -m scripts.seed && uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
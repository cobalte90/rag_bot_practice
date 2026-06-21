FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN useradd --create-home --shell /bin/bash appuser

COPY . .

RUN mkdir -p /app/data/raw /app/data/processed /app/data/index /app/.cache \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "main.py"]

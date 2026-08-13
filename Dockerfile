FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
	fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY app ./app
COPY src ./src
COPY models/topic_classifier/distilbert_topic_classifier \
     ./models/topic_classifier/distilbert_topic_classifier
COPY vector_store ./vector_store
COPY knowledge_base ./knowledge_base

RUN mkdir -p \
    /app/outputs/generated_images \
    /app/outputs/generated_responses \
    /app/outputs/history

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
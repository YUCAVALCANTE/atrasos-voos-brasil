FROM python:3.12-slim

# libgomp é necessária para o LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# O modelo treinado (≈2 MB) é versionado no repositório e entra na imagem.
COPY models ./models
ENV PYTHONUNBUFFERED=1
ENV ATRASOS_MODELO=/app/models/modelo.joblib

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/saude')" || exit 1
CMD ["uvicorn", "atrasos.api:app", "--host", "0.0.0.0", "--port", "8000"]

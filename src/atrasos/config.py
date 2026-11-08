"""Configuração central do projeto: caminhos, período dos dados e divisão temporal."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

VOOS_PARQUET = PROCESSED_DIR / "voos.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
# Pode ser sobrescrito por variável de ambiente (usado na imagem Docker).
MODEL_PATH = Path(os.environ.get("ATRASOS_MODELO", MODELS_DIR / "modelo.joblib"))
METRICS_PATH = REPORTS_DIR / "metricas.json"

# Período baixado da ANAC (inclusive). Os 3 primeiros meses servem só de
# histórico para os atributos "as-of" e não entram no treino.
PRIMEIRO_MES = "2024-01"
ULTIMO_MES = "2026-09"

# Divisão temporal (sem embaralhar: o modelo sempre prevê o futuro).
INICIO_TREINO = "2024-04"
INICIO_VALIDACAO = "2026-01"
INICIO_TESTE = "2026-07"

# Definição de atraso usada pela ANAC e pela literatura: chegada > 15 min após o previsto.
LIMIAR_ATRASO_MIN = 15

# Atributos históricos: janela de meses anteriores usada para calcular taxas de atraso.
JANELA_HISTORICO_MESES = 3
# Força da suavização bayesiana (pseudo-contagem) nas taxas históricas.
SUAVIZACAO = 50

SEED = 42

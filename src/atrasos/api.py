"""API de previsão de atraso.

Rodar localmente:  uvicorn atrasos.api:app --reload
Documentação interativa: http://localhost:8000/docs
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from atrasos import __version__
from atrasos.predicao import carregar_pacote, explicar, prever

app = FastAPI(
    title="Previsão de atrasos de voos — Brasil",
    description="Probabilidade de um voo regular chegar com mais de 15 min de atraso, "
    "treinado com os microdados VRA da ANAC.",
    version=__version__,
)


class Voo(BaseModel):
    empresa: str = Field(..., min_length=3, max_length=3, description="Código ICAO da empresa", examples=["GLO"])
    numero_voo: str = Field(..., max_length=6, examples=["1234"])
    origem: str = Field(..., min_length=4, max_length=4, description="ICAO do aeródromo de origem", examples=["SBGR"])
    destino: str = Field(..., min_length=4, max_length=4, examples=["SBRJ"])
    partida_prevista: datetime = Field(..., examples=["2026-11-20T18:30:00"])
    chegada_prevista: datetime = Field(..., examples=["2026-11-20T19:35:00"])
    tipo_linha: str = Field("N", description="N=nacional, R=regional, I=internacional, C=cargueiro")
    codigo_di: str = Field("0", description="Código de autorização (DI) da ANAC; 0 = voo regular")

    @model_validator(mode="after")
    def checar_horarios(self):
        if self.chegada_prevista <= self.partida_prevista:
            raise ValueError("chegada_prevista precisa ser depois de partida_prevista")
        return self


class Previsao(BaseModel):
    probabilidade_atraso: float
    alerta: bool
    limiar: float
    principais_fatores: list[dict]


@app.get("/saude")
def saude() -> dict:
    try:
        pacote = carregar_pacote()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"status": "ok", "versao_modelo": pacote["versao"], "periodo_treino": pacote["periodo_treino"]}


@app.post("/prever", response_model=list[Previsao])
def prever_voos(voos: list[Voo]) -> list[Previsao]:
    if not voos:
        raise HTTPException(422, "envie ao menos um voo")
    if len(voos) > 1000:
        raise HTTPException(422, "máximo de 1000 voos por requisição")
    pacote = carregar_pacote()
    df = pd.DataFrame([v.model_dump() for v in voos])
    probs = prever(df, pacote)
    fatores = explicar(df, pacote)
    return [
        Previsao(
            probabilidade_atraso=round(float(p), 4),
            alerta=bool(p >= pacote["limiar"]),
            limiar=round(pacote["limiar"], 4),
            principais_fatores=f,
        )
        for p, f in zip(probs, fatores)
    ]

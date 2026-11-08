"""Testa a API ponta a ponta com um modelo minúsculo treinado em dados sintéticos
(o CI não precisa baixar os 400 MB da ANAC)."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from atrasos import api, predicao
from atrasos.atributos import (
    CATEGORICAS,
    FEATURES,
    aplicar_congestionamento,
    aplicar_historico,
    atributos_calendario,
    como_categoricas,
    congestionamento_tipico,
    tabelas_historicas,
)


@pytest.fixture(scope="module")
def pacote():
    rng = np.random.default_rng(0)
    n = 3000
    partida = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 150 * 24, n), unit="h")
    df = pd.DataFrame(
        {
            "empresa": rng.choice(["GLO", "AZU", "TAM"], n),
            "numero_voo": rng.integers(1000, 1050, n).astype(str),
            "origem": rng.choice(["SBGR", "SBSP", "SBKP"], n),
            "destino": rng.choice(["SBRJ", "SBCF"], n),
            "tipo_linha": "N",
            "codigo_di": "0",
            "partida_prevista": partida,
            "chegada_prevista": partida + pd.Timedelta(minutes=70),
        }
    )
    df["atrasado"] = (rng.random(n) < np.where(df["partida_prevista"].dt.hour > 17, 0.4, 0.1)).astype(int)
    df = atributos_calendario(df)
    df = aplicar_congestionamento(df, df)
    tabelas = tabelas_historicas(df)
    df = aplicar_historico(df, tabelas)
    df, cats = como_categoricas(df)
    modelo = lgb.train(
        {"objective": "binary", "verbose": -1, "min_data_in_leaf": 20},
        lgb.Dataset(df[FEATURES], df["atrasado"], categorical_feature=CATEGORICAS),
        num_boost_round=20,
    )
    mes_api = int(df["mes_num"].max()) + 1
    return {
        "versao": "teste",
        "modelo": modelo,
        "features": FEATURES,
        "categorias": cats,
        "tabelas_historicas": {k: v[v["mes_num"] == mes_api] for k, v in tabelas.items()},
        "mes_num_historico": mes_api,
        "congestionamento": congestionamento_tipico(df),
        "limiar": 0.25,
        "periodo_treino": ["2025-01", "2025-05"],
    }


@pytest.fixture
def cliente(pacote, monkeypatch):
    monkeypatch.setattr(api, "carregar_pacote", lambda: pacote)
    monkeypatch.setattr(predicao, "carregar_pacote", lambda: pacote)
    return TestClient(api.app)


VOO = {
    "empresa": "GLO",
    "numero_voo": "1010",
    "origem": "SBGR",
    "destino": "SBRJ",
    "partida_prevista": "2025-06-10T19:00:00",
    "chegada_prevista": "2025-06-10T20:10:00",
}


def test_saude(cliente):
    r = cliente.get("/saude")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_prever_retorna_probabilidade_e_fatores(cliente):
    r = cliente.post(
        "/prever",
        json=[VOO, {**VOO, "partida_prevista": "2025-06-10T06:00:00", "chegada_prevista": "2025-06-10T07:10:00"}],
    )
    assert r.status_code == 200
    corpo = r.json()
    assert len(corpo) == 2
    for item in corpo:
        assert 0 <= item["probabilidade_atraso"] <= 1
        assert len(item["principais_fatores"]) == 5
    # O padrão sintético (voos noturnos atrasam mais) precisa aparecer.
    assert corpo[0]["probabilidade_atraso"] > corpo[1]["probabilidade_atraso"]


def test_empresa_desconhecida_nao_quebra(cliente):
    r = cliente.post("/prever", json=[{**VOO, "empresa": "XYZ", "origem": "SBZZ"}])
    assert r.status_code == 200


def test_chegada_antes_da_partida_e_rejeitada(cliente):
    r = cliente.post("/prever", json=[{**VOO, "chegada_prevista": "2025-06-10T18:00:00"}])
    assert r.status_code == 422

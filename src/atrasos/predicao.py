"""Inferência para voos novos — reaproveita exatamente as funções de atributos do treino."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from atrasos.atributos import aplicar_historico, atributos_calendario, como_categoricas
from atrasos.config import MODEL_PATH


@lru_cache(maxsize=1)
def carregar_pacote(caminho: Path = MODEL_PATH) -> dict:
    if not Path(caminho).exists():
        raise FileNotFoundError(f"Modelo não encontrado em {caminho}. Rode `python -m atrasos.treino`.")
    return joblib.load(caminho)


def montar_atributos(voos: pd.DataFrame, pacote: dict) -> pd.DataFrame:
    df = voos.copy()
    for col in ["empresa", "origem", "destino", "tipo_linha", "codigo_di"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    df["numero_voo"] = df["numero_voo"].astype(str).str.strip().str.lstrip("0").replace("", "0")
    df["partida_prevista"] = pd.to_datetime(df["partida_prevista"])
    df["chegada_prevista"] = pd.to_datetime(df["chegada_prevista"])

    df = atributos_calendario(df)

    # Congestionamento: a API recebe voos isolados, então usa a média histórica de voos
    # programados no aeroporto naquele dia da semana e hora.
    cong = pacote["congestionamento"]
    p = cong["partidas_origem_hora"].rename(columns={"hora": "hora_partida"})
    c = cong["chegadas_destino_hora"].rename(columns={"hora": "hora_chegada"})
    df = df.merge(p, on=["origem", "dia_semana", "hora_partida"], how="left")
    df = df.merge(c, on=["destino", "dia_semana", "hora_chegada"], how="left")
    df[["partidas_origem_hora", "chegadas_destino_hora"]] = df[
        ["partidas_origem_hora", "chegadas_destino_hora"]
    ].fillna(0)

    # Histórico: sempre a janela mais recente disponível no pacote.
    mes_real = df["mes_num"].copy()
    df["mes_num"] = pacote["mes_num_historico"]
    df = aplicar_historico(df, pacote["tabelas_historicas"])
    df["mes_num"] = mes_real.to_numpy()

    df, _ = como_categoricas(df, pacote["categorias"])
    return df


def prever(voos: pd.DataFrame, pacote: dict | None = None) -> np.ndarray:
    pacote = pacote or carregar_pacote()
    X = montar_atributos(voos, pacote)[pacote["features"]]
    return pacote["modelo"].predict(X)


def explicar(voos: pd.DataFrame, pacote: dict | None = None, top: int = 5) -> list[list[dict]]:
    """Contribuições SHAP (em log-odds) dos atributos mais relevantes de cada voo."""
    pacote = pacote or carregar_pacote()
    X = montar_atributos(voos, pacote)[pacote["features"]]
    contrib = pacote["modelo"].predict(X, pred_contrib=True)[:, :-1]  # última coluna = valor base
    saida = []
    for linha, valores in zip(contrib, X.itertuples(index=False)):
        ordem = np.argsort(-np.abs(linha))[:top]
        saida.append(
            [
                {
                    "atributo": pacote["features"][i],
                    "valor": _json(valores[i]),
                    "contribuicao": round(float(linha[i]), 4),
                }
                for i in ordem
            ]
        )
    return saida


def _json(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    return str(v)

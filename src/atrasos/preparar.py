"""Monta a tabela de modelagem (data/processed/features.parquet) a partir dos voos limpos."""

from __future__ import annotations

import pandas as pd

from atrasos.atributos import (
    aplicar_congestionamento,
    aplicar_historico,
    atributos_calendario,
    contexto_operacional,
    tabelas_historicas,
)
from atrasos.config import (
    FEATURES_PARQUET,
    INICIO_TESTE,
    INICIO_TREINO,
    INICIO_VALIDACAO,
    PROCESSED_DIR,
    VOOS_PARQUET,
)


def rotular_divisao(mes: pd.Series) -> pd.Series:
    return pd.Series(
        pd.cut(
            pd.PeriodIndex(mes, freq="M").to_timestamp(),
            bins=[
                pd.Timestamp("1900-01-01"),
                pd.Timestamp(INICIO_TREINO),
                pd.Timestamp(INICIO_VALIDACAO),
                pd.Timestamp(INICIO_TESTE),
                pd.Timestamp("2100-01-01"),
            ],
            labels=["historico", "treino", "validacao", "teste"],
            right=False,
        ),
        index=mes.index,
    ).astype(str)


def montar(voos: pd.DataFrame, malha: pd.DataFrame) -> pd.DataFrame:
    voos = atributos_calendario(voos)
    voos = aplicar_congestionamento(voos, malha)
    voos = aplicar_historico(voos, tabelas_historicas(voos))
    voos = contexto_operacional(voos, voos)
    voos["divisao"] = rotular_divisao(voos["mes"])
    return voos


def main() -> None:
    voos = pd.read_parquet(VOOS_PARQUET)
    malha = pd.read_parquet(
        PROCESSED_DIR / "voos_todos.parquet", columns=["origem", "destino", "partida_prevista", "chegada_prevista"]
    )
    malha = malha.dropna(subset=["partida_prevista", "chegada_prevista"])
    df = montar(voos, malha)
    df.to_parquet(FEATURES_PARQUET, index=False)
    resumo = df.groupby("divisao").agg(
        voos=("atrasado", "size"), taxa_atraso=("atrasado", "mean"), meses=("mes", "nunique")
    )
    print(resumo)


if __name__ == "__main__":
    main()

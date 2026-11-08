"""Testes dos pontos onde é fácil errar: vazamento temporal, suavização e limpeza."""

import numpy as np
import pandas as pd
import pytest

from atrasos.atributos import (
    aplicar_historico,
    atributos_calendario,
    como_categoricas,
    tabelas_historicas,
)
from atrasos.config import SUAVIZACAO
from atrasos.limpeza import filtrar_realizados, limpar


def _voos(n_por_mes: int = 20, meses=("2025-01", "2025-02", "2025-03", "2025-04", "2025-05")) -> pd.DataFrame:
    linhas = []
    for i, mes in enumerate(meses):
        for j in range(n_por_mes):
            partida = pd.Timestamp(f"{mes}-10 08:00") + pd.Timedelta(hours=j % 10)
            linhas.append(
                {
                    "empresa": "GLO" if j % 2 else "AZU",
                    "numero_voo": str(1000 + j % 4),
                    "origem": "SBGR",
                    "destino": "SBRJ",
                    "tipo_linha": "N",
                    "codigo_di": "0",
                    "partida_prevista": partida,
                    "chegada_prevista": partida + pd.Timedelta(minutes=60),
                    # mês i: só GLO atrasa, e só a partir do 3º mês
                    "atrasado": int(j % 2 == 1 and i >= 2),
                }
            )
    return pd.DataFrame(linhas)


def test_historico_usa_somente_meses_anteriores():
    df = atributos_calendario(_voos())
    out = aplicar_historico(df, tabelas_historicas(df))
    glo = out[out["empresa"] == "GLO"].groupby("mes")["hist_empresa_taxa"].first()
    # Em 2025-03 a GLO começa a atrasar, mas o histórico daquele mês só vê jan/fev (zero atrasos).
    assert glo["2025-03"] == pytest.approx(0.0)
    # Em 2025-04 o histórico já enxerga março.
    assert glo["2025-04"] > 0


def test_primeiro_mes_cai_na_media_global_sem_historico():
    df = atributos_calendario(_voos())
    out = aplicar_historico(df, tabelas_historicas(df))
    jan = out[out["mes"] == "2025-01"]
    assert (jan["hist_empresa_log_n"] == 0).all()


def test_suavizacao_puxa_para_media_global():
    df = atributos_calendario(_voos(n_por_mes=4))
    out = aplicar_historico(df, tabelas_historicas(df))
    abr_glo = out[(out["mes"] == "2025-04") & (out["empresa"] == "GLO")].iloc[0]
    n_glo = 2 * 3  # 2 voos GLO/mês nos 3 meses anteriores
    atr_glo = 2  # só março atrasou
    p_global = 2 / 12  # 2 atrasos em 12 voos na janela
    esperado = (atr_glo + SUAVIZACAO * p_global) / (n_glo + SUAVIZACAO)
    assert abr_glo["hist_empresa_taxa"] == pytest.approx(esperado)


def test_calendario_e_feriado():
    df = pd.DataFrame(
        {
            "origem": ["SBGR"],
            "destino": ["SBSP"],
            "partida_prevista": [pd.Timestamp("2025-12-25 10:00")],
            "chegada_prevista": [pd.Timestamp("2025-12-25 11:05")],
        }
    )
    out = atributos_calendario(df)
    assert out.loc[0, "feriado"] == 1
    assert out.loc[0, "tempo_bloco_previsto_min"] == 65
    assert out.loc[0, "rota"] == "SBGR-SBSP"
    assert out.loc[0, "dia_semana"] == 3  # quinta-feira


def test_categoria_desconhecida_vira_nan():
    treino = pd.DataFrame({c: ["A"] for c in ["empresa", "origem", "destino", "rota", "tipo_linha", "codigo_di"]})
    _, cats = como_categoricas(treino)
    novo = treino.assign(empresa="ZZZ")
    out, _ = como_categoricas(novo, cats)
    assert pd.isna(out.loc[0, "empresa"])
    assert out.loc[0, "origem"] == "A"


def test_limpeza_calcula_atraso_e_filtra():
    bruto = pd.DataFrame(
        {
            "empresa": ["glo", "AZU", "TAM"],
            "numero_voo": ["0123", "45", "7"],
            "codigo_di": ["0", "0", "0"],
            "tipo_linha": ["N", "N", "N"],
            "origem": ["SBGR", "SBKP", "SBBR"],
            "destino": ["SBRJ", "SBCF", "SBSV"],
            "partida_prevista": ["2025-01-01 10:00:00"] * 3,
            "partida_real": ["2025-01-01 10:20:00", None, "2025-01-01 10:00:00"],
            "chegada_prevista": ["2025-01-01 11:00:00"] * 3,
            "chegada_real": ["2025-01-01 11:30:00", None, "2025-01-01 10:55:00"],
            "situacao": ["REALIZADO", "CANCELADO", "REALIZADO"],
            "codigo_justificativa": [np.nan] * 3,
        }
    )
    limpo = limpar(bruto)
    assert limpo.loc[0, "empresa"] == "GLO"
    assert limpo.loc[0, "numero_voo"] == "123"
    assert limpo.loc[0, "atraso_chegada_min"] == 30
    assert limpo["atrasado"].tolist() == [1, 0, 0]
    realizados = filtrar_realizados(limpo)
    assert realizados["empresa"].tolist() == ["GLO", "TAM"]

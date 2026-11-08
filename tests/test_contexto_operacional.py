"""O cenário operacional só pode enxergar eventos que já aconteceram no instante da previsão."""

import numpy as np
import pandas as pd
import pytest

from atrasos.atributos import HORIZONTE_H, JANELA_OPERACIONAL_H, contexto_operacional


def _evento(origem, partida_real, atraso, destino="SBRJ", empresa="GLO"):
    partida_real = pd.Timestamp(partida_real)
    return {
        "origem": origem,
        "destino": destino,
        "empresa": empresa,
        "partida_prevista": partida_real - pd.Timedelta(minutes=atraso),
        "partida_real": partida_real,
        "chegada_real": partida_real + pd.Timedelta(hours=1),
        "atraso_partida_min": atraso,
        "atraso_chegada_min": atraso,
    }


def test_so_usa_eventos_dentro_da_janela_antes_do_horizonte():
    alvo_partida = pd.Timestamp("2025-05-01 18:00")
    t_prev = alvo_partida - pd.Timedelta(hours=HORIZONTE_H)  # 16:00
    eventos = pd.DataFrame(
        [
            _evento("SBGR", t_prev - pd.Timedelta(hours=JANELA_OPERACIONAL_H + 1), 999),  # velho demais
            _evento("SBGR", t_prev - pd.Timedelta(hours=1), 30),  # conta
            _evento("SBGR", t_prev - pd.Timedelta(minutes=10), 50),  # conta
            _evento("SBGR", t_prev + pd.Timedelta(minutes=5), 500),  # futuro: NÃO pode contar
            _evento("SBKP", t_prev - pd.Timedelta(hours=1), 400),  # outro aeroporto
        ]
    )
    alvo = pd.DataFrame(
        {"origem": ["SBGR"], "destino": ["SBCF"], "empresa": ["AZU"], "partida_prevista": [alvo_partida]}
    )
    out = contexto_operacional(alvo, eventos)
    assert out.loc[0, "op_origem_n_partidas"] == 2
    assert out.loc[0, "op_origem_atraso_partida_medio"] == pytest.approx(40)
    assert out.loc[0, "op_destino_n_chegadas"] == 0
    assert np.isnan(out.loc[0, "op_destino_atraso_chegada_medio"])

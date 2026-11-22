"""Testes da projeção: preenchimento da pandemia, formato das previsões e ausência de vazamento."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from atrasos.projecao import MODELOS, PANDEMIA, preencher_pandemia, prever, validar


def _serie(inicio="2015-01", fim="2026-09", nivel=0.15, amplitude=0.04, seed=0) -> pd.Series:
    idx = pd.period_range(inicio, fim, freq="M")
    rng = np.random.default_rng(seed)
    v = nivel + amplitude * np.sin(2 * np.pi * (idx.month - 1) / 12) + rng.normal(0, 0.005, len(idx))
    return pd.Series(v, index=idx)


def test_preenchimento_troca_so_a_pandemia_e_preserva_a_sazonalidade():
    s = _serie()
    choque = s.copy()
    gap = (s.index >= PANDEMIA[0]) & (s.index <= PANDEMIA[1])
    choque[gap] = 0.9  # valores absurdos durante a pandemia
    cheio = preencher_pandemia(choque)
    assert (cheio[~gap] == choque[~gap]).all()
    assert np.abs(cheio[gap] - s[gap]).max() < 0.03  # volta para perto do padrão verdadeiro


@pytest.mark.parametrize("nome", list(MODELOS))
def test_modelos_devolvem_h_valores_validos(nome):
    s = preencher_pandemia(_serie())
    p = prever(nome, s, 27, taxa=True)
    assert p.shape == (27,)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_validacao_nao_ve_o_futuro():
    s = _serie()
    marcado = s.copy()
    marcado[marcado.index > pd.Period("2024-09", freq="M")] = 0.99  # futuro "envenenado"
    a = validar(s, taxa=True)
    b = validar(marcado, taxa=True)
    # previsões feitas em cortes até 2024-09 não podem mudar quando só o futuro muda
    cols = ["corte", "modelo", "h", "previsto"]
    ok = a[a["mes"] <= "2024-09"][cols].merge(b[b["mes"] <= "2024-09"][cols], on=cols[:3])
    assert np.allclose(ok["previsto_x"], ok["previsto_y"])

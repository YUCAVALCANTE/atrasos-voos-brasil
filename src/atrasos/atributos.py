"""Engenharia de atributos.

Regra de ouro: todo atributo precisa ser conhecido **antes da partida**. Por isso:

* atributos de calendário e de malha vêm só dos horários *previstos*;
* congestionamento = quantos voos estão *programados* no mesmo aeroporto e hora;
* taxas históricas usam apenas os meses **anteriores** ao mês do voo (atributos "as-of"),
  com suavização bayesiana para rotas e voos com pouco histórico.

O mesmo código é usado no treino e na API, evitando divergência treino/produção.
"""

from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from atrasos.config import JANELA_HISTORICO_MESES, SUAVIZACAO

CATEGORICAS = ["empresa", "origem", "destino", "rota", "tipo_linha", "codigo_di"]

# nome -> chaves de agrupamento das taxas históricas
CHAVES_HISTORICO: dict[str, list[str]] = {
    "empresa": ["empresa"],
    "origem": ["origem"],
    "destino": ["destino"],
    "rota": ["origem", "destino"],
    "voo": ["empresa", "numero_voo"],
    "empresa_origem": ["empresa", "origem"],
    "origem_hora": ["origem", "hora_partida"],
}

NUMERICAS = [
    "hora_partida",
    "hora_chegada",
    "dia_semana",
    "mes_do_ano",
    "dia_do_mes",
    "feriado",
    "dias_ate_feriado",
    "tempo_bloco_previsto_min",
    "partidas_origem_hora",
    "chegadas_destino_hora",
    *[f"hist_{n}_taxa" for n in CHAVES_HISTORICO],
    *[f"hist_{n}_log_n" for n in CHAVES_HISTORICO],
]

FEATURES = CATEGORICAS + NUMERICAS  # cenário "planejamento" (véspera): é o que a API serve

# Cenário "operacional": previsão feita HORIZONTE_H horas antes da partida, já sabendo como
# a operação está rodando naquele momento (só eventos que JÁ aconteceram até lá).
HORIZONTE_H = 2
JANELA_OPERACIONAL_H = 3
OPERACIONAIS = [
    "op_origem_atraso_partida_medio",
    "op_origem_n_partidas",
    "op_destino_atraso_chegada_medio",
    "op_destino_n_chegadas",
    "op_empresa_atraso_partida_medio",
    "op_empresa_n_partidas",
]
FEATURES_OPERACIONAL = FEATURES + OPERACIONAIS


def _janela_movel(alvo_chave, alvo_t, ev_chave, ev_t, ev_valor, janela) -> tuple[np.ndarray, np.ndarray]:
    """Para cada alvo, média e contagem de eventos da mesma chave com tempo em (t - janela, t]."""
    media = np.full(len(alvo_t), np.nan)
    contagem = np.zeros(len(alvo_t))
    ev = pd.DataFrame({"k": ev_chave, "t": ev_t, "v": ev_valor}).sort_values(["k", "t"])
    grupos_ev = {k: g for k, g in ev.groupby("k", sort=False)}
    alvos = pd.DataFrame({"k": alvo_chave, "t": np.asarray(alvo_t), "pos": np.arange(len(alvo_t))})
    for k, grupo_alvo in alvos.groupby("k", sort=False):
        grupo_ev = grupos_ev.get(k)
        if grupo_ev is None:
            continue
        tempos = grupo_ev["t"].to_numpy()
        acumulado = np.concatenate([[0.0], np.cumsum(grupo_ev["v"].to_numpy())])
        t_alvo = grupo_alvo["t"].to_numpy()
        fim = np.searchsorted(tempos, t_alvo, side="right")
        ini = np.searchsorted(tempos, t_alvo - janela.to_timedelta64(), side="right")
        n = fim - ini
        soma = acumulado[fim] - acumulado[ini]
        pos = grupo_alvo["pos"].to_numpy()
        contagem[pos] = n
        media[pos] = np.where(n > 0, soma / np.maximum(n, 1), np.nan)
    return media, contagem


def contexto_operacional(alvo: pd.DataFrame, eventos: pd.DataFrame) -> pd.DataFrame:
    """Atrasos observados nas últimas JANELA_OPERACIONAL_H horas antes do instante de previsão
    (partida prevista - HORIZONTE_H). `eventos` = voos realizados com horários reais."""
    out = alvo.copy()
    t_previsao = out["partida_prevista"] - pd.Timedelta(hours=HORIZONTE_H)
    janela = pd.Timedelta(hours=JANELA_OPERACIONAL_H)
    ev = eventos.dropna(subset=["partida_real", "chegada_real"])
    atr_p = ev["atraso_partida_min"].clip(-60, 600)
    atr_c = ev["atraso_chegada_min"].clip(-60, 600)

    specs = [
        ("origem", "origem", "partida_real", atr_p, "op_origem_atraso_partida_medio", "op_origem_n_partidas"),
        ("destino", "destino", "chegada_real", atr_c, "op_destino_atraso_chegada_medio", "op_destino_n_chegadas"),
        ("empresa", "empresa", "partida_real", atr_p, "op_empresa_atraso_partida_medio", "op_empresa_n_partidas"),
    ]
    for col_alvo, col_ev, col_t, valores, nome_media, nome_n in specs:
        media, n = _janela_movel(
            out[col_alvo].astype(str).to_numpy(),
            t_previsao,
            ev[col_ev].astype(str).to_numpy(),
            ev[col_t].to_numpy(),
            valores.to_numpy(),
            janela,
        )
        out[nome_media] = media
        out[nome_n] = n
    return out


def mes_para_num(mes: pd.Series) -> pd.Series:
    """'2025-03' -> índice inteiro de meses (ano*12 + mês), facilita deslocar janelas."""
    p = pd.PeriodIndex(mes, freq="M")
    return pd.Series(p.year * 12 + p.month - 1, index=mes.index)


@lru_cache(maxsize=1)
def _feriados(ano_min: int, ano_max: int) -> np.ndarray:
    datas = holidays.Brazil(years=range(ano_min - 1, ano_max + 2))
    return np.array(sorted(pd.Timestamp(d) for d in datas), dtype="datetime64[ns]")


def atributos_calendario(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pp, cp = out["partida_prevista"], out["chegada_prevista"]
    out["hora_partida"] = pp.dt.hour.astype("int16")
    out["hora_chegada"] = cp.dt.hour.astype("int16")
    out["dia_semana"] = pp.dt.day_of_week.astype("int16")
    out["mes_do_ano"] = pp.dt.month.astype("int16")
    out["dia_do_mes"] = pp.dt.day.astype("int16")
    out["tempo_bloco_previsto_min"] = (cp - pp).dt.total_seconds() / 60
    out["rota"] = out["origem"] + "-" + out["destino"]
    out["mes"] = pp.dt.to_period("M").astype(str)
    out["mes_num"] = mes_para_num(out["mes"])

    datas = pp.dt.normalize().to_numpy()
    feriados = _feriados(int(pp.dt.year.min()), int(pp.dt.year.max()))
    idx = np.searchsorted(feriados, datas)
    anterior = feriados[np.clip(idx - 1, 0, len(feriados) - 1)]
    proximo = feriados[np.clip(idx, 0, len(feriados) - 1)]
    dist = np.minimum(np.abs(datas - anterior), np.abs(proximo - datas)) / np.timedelta64(1, "D")
    out["dias_ate_feriado"] = np.minimum(dist, 15).astype("int16")
    out["feriado"] = (out["dias_ate_feriado"] == 0).astype("int8")
    return out


def contar_congestionamento(malha: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Voos programados por aeroporto e hora (inclui cancelados: é a malha planejada)."""
    part = (
        malha.assign(slot=malha["partida_prevista"].dt.floor("h"))
        .groupby(["origem", "slot"])
        .size()
        .rename("partidas_origem_hora")
        .reset_index()
    )
    cheg = (
        malha.assign(slot=malha["chegada_prevista"].dt.floor("h"))
        .groupby(["destino", "slot"])
        .size()
        .rename("chegadas_destino_hora")
        .reset_index()
    )
    return part, cheg


def aplicar_congestionamento(df: pd.DataFrame, malha: pd.DataFrame) -> pd.DataFrame:
    part, cheg = contar_congestionamento(malha)
    out = df.assign(
        _slot_p=df["partida_prevista"].dt.floor("h"),
        _slot_c=df["chegada_prevista"].dt.floor("h"),
    )
    out = out.merge(part, left_on=["origem", "_slot_p"], right_on=["origem", "slot"], how="left").drop(columns="slot")
    out = out.merge(cheg, left_on=["destino", "_slot_c"], right_on=["destino", "slot"], how="left").drop(columns="slot")
    out[["partidas_origem_hora", "chegadas_destino_hora"]] = out[
        ["partidas_origem_hora", "chegadas_destino_hora"]
    ].fillna(0)
    return out.drop(columns=["_slot_p", "_slot_c"])


def congestionamento_tipico(malha: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Média de voos programados por (aeroporto, dia da semana, hora) — usado pela API,
    que recebe um voo isolado e não a malha inteira do dia."""
    part, cheg = contar_congestionamento(malha)
    out = {}
    for nome, tab, aer in [("partidas_origem_hora", part, "origem"), ("chegadas_destino_hora", cheg, "destino")]:
        tab = tab.assign(dia_semana=tab["slot"].dt.day_of_week, hora=tab["slot"].dt.hour)
        dias = tab.groupby([aer, "dia_semana"])["slot"].apply(lambda s: s.dt.normalize().nunique())
        soma = tab.groupby([aer, "dia_semana", "hora"])[nome].sum()
        out[nome] = (soma / dias.reindex(soma.index.droplevel("hora")).to_numpy()).rename(nome).reset_index()
    return out


def tabelas_historicas(voos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Para cada chave e cada mês M, soma voos e atrasos dos meses M-1..M-JANELA.

    Retorna tabelas indexadas por (chaves..., mes_num) prontas para merge.
    """
    base = voos[["mes_num", "atrasado", *{c for ks in CHAVES_HISTORICO.values() for c in ks}]]
    tabelas: dict[str, pd.DataFrame] = {}
    for nome, chaves in {**CHAVES_HISTORICO, "_global": []}.items():
        agg = base.groupby([*chaves, "mes_num"], observed=True)["atrasado"].agg(n="size", atr="sum").reset_index()
        deslocados = [agg.assign(mes_num=agg["mes_num"] + lag) for lag in range(1, JANELA_HISTORICO_MESES + 1)]
        tabelas[nome] = (
            pd.concat(deslocados).groupby([*chaves, "mes_num"], observed=True)[["n", "atr"]].sum().reset_index()
        )
    return tabelas


def aplicar_historico(df: pd.DataFrame, tabelas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    glob = tabelas["_global"].assign(p_global=lambda t: t["atr"] / t["n"])[["mes_num", "p_global"]]
    out = out.merge(glob, on="mes_num", how="left")
    out["p_global"] = out["p_global"].fillna(glob["p_global"].mean() if len(glob) else 0.2)
    for nome, chaves in CHAVES_HISTORICO.items():
        tab = tabelas[nome].rename(columns={"n": "_n", "atr": "_atr"})
        out = out.merge(tab, on=[*chaves, "mes_num"], how="left")
        n = out.pop("_n").fillna(0)
        atr = out.pop("_atr").fillna(0)
        out[f"hist_{nome}_taxa"] = (atr + SUAVIZACAO * out["p_global"]) / (n + SUAVIZACAO)
        out[f"hist_{nome}_log_n"] = np.log1p(n)
    return out.drop(columns="p_global")


def como_categoricas(df: pd.DataFrame, categorias: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Converte colunas categóricas com um vocabulário fixo (o do treino).
    Valores nunca vistos viram NaN — o LightGBM trata como 'desconhecido'."""
    out = df.copy()
    if categorias is None:
        categorias = {c: sorted(out[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAS}
    for c in CATEGORICAS:
        valores = out[c].astype(str)
        valores = valores.where(valores.isin(categorias[c]))
        out[c] = pd.Categorical(valores, categories=categorias[c])
    return out, categorias

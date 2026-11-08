"""Agrega os voos em tabelas pequenas (JSON) para o dashboard interativo.

Cada tabela traz todas as combinações de filtro já calculadas — empresa x aeroporto de
origem x ano, cada um com a opção "Todas/Todos" — para os filtros responderem na hora,
sem processar 2,5 milhões de linhas no navegador.
"""

from __future__ import annotations

import itertools
import json

import pandas as pd

from atrasos.config import METRICS_PATH, PROCESSED_DIR, REPORTS_DIR

SAIDA = REPORTS_DIR / "dashboard"
TODAS, TODOS = "Todas", "Todos"
N_AEROPORTOS = 20
N_ROTAS = 12

EMPRESAS = {"GLO": "Gol", "TAM": "LATAM", "AZU": "Azul", "PTB": "Voepass", "ACN": "Azul Conecta", "SID": "Sideral"}
AEROPORTOS = {
    "SBGR": "Guarulhos (SP)",
    "SBSP": "Congonhas (SP)",
    "SBBR": "Brasília (DF)",
    "SBKP": "Viracopos (SP)",
    "SBGL": "Galeão (RJ)",
    "SBRJ": "Santos Dumont (RJ)",
    "SBCF": "Confins (MG)",
    "SBPA": "Porto Alegre (RS)",
    "SBRF": "Recife (PE)",
    "SBSV": "Salvador (BA)",
    "SBFZ": "Fortaleza (CE)",
    "SBCT": "Curitiba (PR)",
    "SBFL": "Florianópolis (SC)",
    "SBBE": "Belém (PA)",
    "SBEG": "Manaus (AM)",
    "SBGO": "Goiânia (GO)",
    "SBCY": "Cuiabá (MT)",
    "SBVT": "Vitória (ES)",
    "SBMO": "Maceió (AL)",
    "SBSG": "Natal (RN)",
    "SBSL": "São Luís (MA)",
    "SBJP": "João Pessoa (PB)",
    "SBPS": "Porto Seguro (BA)",
    "SBCG": "Campo Grande (MS)",
    "SBUL": "Uberlândia (MG)",
    "SBNF": "Navegantes (SC)",
    "SBFI": "Foz do Iguaçu (PR)",
    "SBTE": "Teresina (PI)",
    "SBAR": "Aracaju (SE)",
    "SBPJ": "Palmas (TO)",
    "SBLO": "Londrina (PR)",
    "SBJV": "Joinville (SC)",
}
DIAS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
NOMES_ATRIBUTOS = {
    "hist_voo_taxa": "Histórico do número de voo",
    "hist_origem_hora_taxa": "Histórico do aeroporto naquela hora",
    "hist_rota_taxa": "Histórico da rota",
    "hist_empresa_origem_taxa": "Histórico da empresa no aeroporto",
    "hist_origem_taxa": "Histórico do aeroporto de origem",
    "hist_destino_taxa": "Histórico do aeroporto de destino",
    "hist_empresa_taxa": "Histórico da empresa",
    "hora_partida": "Hora programada de partida",
    "hora_chegada": "Hora programada de chegada",
    "mes_do_ano": "Mês do ano",
    "dia_do_mes": "Dia do mês",
    "dia_semana": "Dia da semana",
    "dias_ate_feriado": "Distância até feriado",
    "feriado": "É feriado",
    "tempo_bloco_previsto_min": "Duração prevista do voo",
    "partidas_origem_hora": "Partidas programadas na mesma hora",
    "chegadas_destino_hora": "Chegadas programadas na mesma hora",
    "rota": "Rota",
    "origem": "Aeroporto de origem",
    "destino": "Aeroporto de destino",
    "empresa": "Empresa",
    "op_origem_atraso_partida_medio": "Atraso atual no aeroporto de origem",
    "op_destino_atraso_chegada_medio": "Atraso atual no aeroporto de destino",
    "op_empresa_atraso_partida_medio": "Atraso atual na rede da empresa",
    "op_origem_n_partidas": "Partidas recentes na origem",
    "op_destino_n_chegadas": "Chegadas recentes no destino",
    "op_empresa_n_partidas": "Partidas recentes da empresa",
}


def variantes(df: pd.DataFrame, chaves_extra: list[str]) -> pd.DataFrame:
    """Replica o DataFrame para cada combinação "valor real" / "Todas" das dimensões de filtro."""
    partes = []
    for emp_tudo, aer_tudo, ano_tudo in itertools.product([False, True], repeat=3):
        parte = df.copy()
        if emp_tudo:
            parte["empresa"] = TODAS
        if aer_tudo:
            parte["aeroporto"] = TODOS
        if ano_tudo:
            parte["ano"] = TODOS
        partes.append(parte)
    out = pd.concat(partes, ignore_index=True)
    return out[
        [
            "empresa",
            "aeroporto",
            "ano",
            *chaves_extra,
            *[c for c in df.columns if c not in ("empresa", "aeroporto", "ano", *chaves_extra)],
        ]
    ]


def preparar_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["empresa", "origem", "destino", "mes", "partida_prevista", "atrasado", "atraso_chegada_min"]
    voos = pd.read_parquet(PROCESSED_DIR / "voos.parquet", columns=cols)
    todos = pd.read_parquet(PROCESSED_DIR / "voos_todos.parquet", columns=["empresa", "origem", "mes", "cancelado"])
    meses = voos["mes"].value_counts()[lambda s: s > 10_000].index  # só meses completos
    voos, todos = voos[voos["mes"].isin(meses)].copy(), todos[todos["mes"].isin(meses)].copy()

    # Escopo: voos que partem de aeroportos brasileiros, operados por empresas brasileiras.
    top_aer = voos["origem"].value_counts().head(N_AEROPORTOS).index
    for d in (voos, todos):
        d["ano"] = d["mes"].str[:4]
        d["aeroporto"] = d["origem"].where(d["origem"].isin(top_aer), "Outros")
    voos = voos[voos["empresa"].isin(EMPRESAS)]
    todos = todos[todos["empresa"].isin(EMPRESAS)]
    voos["atrasado_60"] = (voos["atraso_chegada_min"] > 60).astype("int8")
    voos["atraso_se_atrasado"] = voos["atraso_chegada_min"].where(voos["atrasado"] == 1)
    return voos, todos


def agregar(voos: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    base = variantes(
        voos[["empresa", "aeroporto", "ano", *chaves, "atrasado", "atrasado_60", "atraso_se_atrasado"]], chaves
    )
    g = base.groupby(["empresa", "aeroporto", "ano", *chaves], observed=True)
    return g.agg(
        voos_realizados=("atrasado", "size"),
        taxa_atraso=("atrasado", "mean"),
        taxa_atraso_60=("atrasado_60", "mean"),
        atraso_medio_min=("atraso_se_atrasado", "mean"),
    ).reset_index()


def rotular(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["empresa_nome"] = df["empresa"].map(EMPRESAS).fillna(df["empresa"])
    df["aeroporto_nome"] = df["aeroporto"].map(AEROPORTOS).fillna(df["aeroporto"])
    return df


def arredondar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(4 if c.startswith("taxa") or c in ("roc_auc", "pr_auc", "brier") else 1)
    return df


def salvar(nome: str, df: pd.DataFrame) -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    caminho = SAIDA / f"{nome}.json"
    df = arredondar(df)
    caminho.write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"{nome}: {len(df):,} linhas, {caminho.stat().st_size / 1e3:.0f} KB")


def main() -> None:
    voos, todos = preparar_base()

    # Indicadores por combinação de filtro (inclui cancelamento, que vem da malha completa).
    kpis = agregar(voos, [])
    canc = variantes(todos[["empresa", "aeroporto", "ano", "cancelado"]], [])
    canc = (
        canc.groupby(["empresa", "aeroporto", "ano"])
        .agg(voos_programados=("cancelado", "size"), taxa_cancelamento=("cancelado", "mean"))
        .reset_index()
    )
    kpis = kpis.merge(canc, on=["empresa", "aeroporto", "ano"], how="left")
    kpis.insert(0, "chave", kpis["empresa"] + "|" + kpis["aeroporto"] + "|" + kpis["ano"])
    salvar("indicadores", rotular(kpis))

    mensal = agregar(voos, ["mes"])
    canc_m = variantes(todos[["empresa", "aeroporto", "ano", "mes", "cancelado"]], ["mes"])
    canc_m = canc_m[canc_m["ano"] == TODOS].groupby(["empresa", "aeroporto", "mes"])["cancelado"].mean()
    mensal = mensal[mensal["ano"] == TODOS].drop(columns="ano")
    mensal = mensal.merge(
        canc_m.rename("taxa_cancelamento").reset_index(), on=["empresa", "aeroporto", "mes"], how="left"
    )
    salvar("mensal", mensal[["empresa", "aeroporto", "mes", "voos_realizados", "taxa_atraso", "taxa_cancelamento"]])

    voos["hora"] = voos["partida_prevista"].dt.hour
    salvar(
        "por_hora", agregar(voos, ["hora"])[["empresa", "aeroporto", "ano", "hora", "voos_realizados", "taxa_atraso"]]
    )

    voos["dia_semana"] = voos["partida_prevista"].dt.day_of_week.map(dict(enumerate(DIAS)))
    dia = agregar(voos, ["dia_semana"])
    dia["ordem"] = dia["dia_semana"].map({d: i for i, d in enumerate(DIAS)})
    salvar(
        "por_dia_semana", dia[["empresa", "aeroporto", "ano", "dia_semana", "ordem", "voos_realizados", "taxa_atraso"]]
    )

    voos["rota"] = voos["origem"] + " → " + voos["destino"]
    rotas = agregar(voos, ["rota"])
    rotas = rotas[rotas["voos_realizados"] >= 200]
    rotas = rotas.sort_values("voos_realizados", ascending=False).groupby(["empresa", "aeroporto", "ano"]).head(40)
    rotas = rotas.sort_values("taxa_atraso", ascending=False).groupby(["empresa", "aeroporto", "ano"]).head(N_ROTAS)
    rotas["destino_nome"] = rotas["rota"].str.split(" → ").str[1].map(AEROPORTOS)
    rotas["chave"] = rotas["empresa"] + "|" + rotas["aeroporto"] + "|" + rotas["ano"] + "|" + rotas["rota"]
    salvar(
        "rotas",
        rotas[
            [
                "chave",
                "empresa",
                "aeroporto",
                "ano",
                "rota",
                "destino_nome",
                "voos_realizados",
                "taxa_atraso",
                "atraso_medio_min",
            ]
        ],
    )

    # Filtros: opções com rótulo
    opcoes = pd.concat(
        [
            pd.DataFrame(
                {"filtro": "empresa", "valor": [TODAS, *EMPRESAS], "rotulo": ["Todas as empresas", *EMPRESAS.values()]}
            ),
            pd.DataFrame(
                {
                    "filtro": "aeroporto",
                    "valor": [
                        TODOS,
                        *sorted(a for a in kpis["aeroporto"].unique() if a not in (TODOS, "Outros")),
                        "Outros",
                    ],
                }
            ).assign(
                rotulo=lambda d: (
                    d["valor"]
                    .map(
                        {
                            TODOS: "Todos os aeroportos",
                            "Outros": "Outros aeroportos",
                            **{k: f"{v} · {k}" for k, v in AEROPORTOS.items()},
                        }
                    )
                    .fillna(d["valor"])
                )
            ),
            pd.DataFrame({"filtro": "ano", "valor": [TODOS, *sorted(voos["ano"].unique())]}).assign(
                rotulo=lambda d: d["valor"].replace({TODOS: "Jan/2024 – set/2026"})
            ),
        ],
        ignore_index=True,
    )
    salvar("filtros", opcoes)

    # Modelo
    metricas = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    nomes = {
        "baseline_historico_voo": "Linha de base: histórico do voo",
        "regressao_logistica": "Regressão logística",
        "lightgbm": "LightGBM · planejamento (véspera)",
        "lightgbm_operacional": "LightGBM · operacional (2h antes)",
    }
    linhas = [
        {
            "modelo": nomes[k],
            "ordem": i,
            **{m: v[m] for m in ("roc_auc", "pr_auc", "brier", "precisao_top10", "lift_top10")},
        }
        for i, (k, v) in enumerate((k, r["teste"]) for k, r in metricas["modelos"].items())
    ]
    salvar("modelo_metricas", pd.DataFrame(linhas))

    pred = pd.read_parquet(REPORTS_DIR / "predicoes_teste.parquet")
    decis = []
    for col, cen in [("p_lightgbm", "Planejamento"), ("p_lightgbm_operacional", "Operacional")]:
        dec = pd.qcut(pred[col].rank(method="first"), 10, labels=range(1, 11))
        t = (
            pred.groupby(dec, observed=True)
            .agg(taxa_real=("atrasado", "mean"), prob_media=(col, "mean"))
            .reset_index(names="decil")
        )
        decis.append(t.assign(cenario=cen, decil=t["decil"].astype(int)))
    salvar("modelo_decis", pd.concat(decis)[["cenario", "decil", "taxa_real", "prob_media"]])

    imp = pd.read_csv(REPORTS_DIR / "importancia_atributos_operacional.csv")
    imp["participacao"] = imp["ganho"] / imp["ganho"].sum()
    imp["atributo_nome"] = imp["atributo"].map(NOMES_ATRIBUTOS).fillna(imp["atributo"])
    imp["grupo"] = imp["atributo"].map(
        lambda a: (
            "Operação em tempo real"
            if a.startswith("op_")
            else "Histórico"
            if a.startswith("hist_")
            else "Malha e calendário"
        )
    )
    salvar("modelo_importancia", imp.head(12)[["atributo", "atributo_nome", "grupo", "participacao"]])


if __name__ == "__main__":
    main()

"""Projeção 2027–2028: compara algoritmos de séries temporais com validação no tempo
e projeta cada série com o vencedor, com faixa de incerteza empírica.

Protocolo
---------
1. Séries mensais de voos domésticos (serie_mensal.py), baixadas desde jan/2015.
2. Série comparável desde mar/2018 (ver INICIO_SERIE). Pandemia (fev/2020 a dez/2021):
   os valores observados são substituídos por uma
   estimativa (padrão sazonal dos outros anos + nível interpolado entre jan/2020 e
   jan/2022), para que os modelos não aprendam o choque como padrão.
3. Validação com origem móvel: em cada corte, cada modelo vê só o passado e prevê os
   24 meses seguintes. Sete cortes trimestrais, de 2023-03 a 2024-09.
4. Vence, por série, o menor erro absoluto médio (MAE) nos horizontes 1 a 24.
5. A faixa de incerteza é o percentil 80 dos erros absolutos do vencedor na validação,
   por bloco de 6 meses de horizonte, aplicada em torno da projeção central.

Saída: reports/projecao/{validacao.csv, projecao.csv, resumo.json} e reports/dashboard/projecao*.json
"""

from __future__ import annotations

import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from atrasos.config import PROCESSED_DIR, REPORTS_DIR, SEED

warnings.filterwarnings("ignore")

SAIDA = REPORTS_DIR / "projecao"
FIM_PROJECAO = pd.Period("2028-12", freq="M")
# A ANAC mudou a forma de registrar os horários algumas vezes: até set/2015 o atraso real
# quase sempre aparece como zero e de out/2015 a jan/2018 a taxa salta para 30–47%. A série
# comparável com a atual começa em fev/2018; usamos meses completos a partir de mar/2018.
INICIO_SERIE = pd.Period("2018-03", freq="M")
# fev/2020 veio com arquivo inconsistente na ANAC (271 mil voos, 0% de atraso) e mar/2020 a
# dez/2021 é a pandemia: os dois viram uma lacuna preenchida.
PANDEMIA = (pd.Period("2020-02", freq="M"), pd.Period("2021-12", freq="M"))
HORIZONTE_VALIDACAO = 24
CORTES = list(pd.period_range("2023-03", "2024-09", freq="M")[::3])

# (escopo, métrica, é taxa?)
SERIES = [
    ("Brasil", "taxa_atraso", True),
    ("Brasil", "taxa_cancelamento", True),
    ("Brasil", "voos_programados", False),
    ("Gol", "taxa_atraso", True),
    ("LATAM", "taxa_atraso", True),
    ("Azul", "taxa_atraso", True),
]

NOMES = {
    "ingenuo_sazonal": "Ingênuo sazonal",
    "media_sazonal_2a": "Média sazonal de 2 anos",
    "holt_winters": "Holt-Winters amortecido",
    "sarima": "SARIMA",
    "theta": "Theta",
    "regressao": "Regressão tendência + sazonalidade",
    "lightgbm": "LightGBM com defasagens",
}


# ---------------------------------------------------------------- preparação


def preencher_pandemia(s: pd.Series) -> pd.Series:
    """Troca os meses da pandemia por padrão sazonal + nível interpolado (só com dados de `s`)."""
    s = s.copy()
    gap = (s.index >= PANDEMIA[0]) & (s.index <= PANDEMIA[1])
    if not gap.any():
        return s
    fora = s[~gap].dropna()
    anos = fora.groupby(fora.index.year).transform("mean")
    sazonal = (fora - anos).groupby(fora.index.month).mean()
    saz = pd.Series(s.index.month.map(sazonal).astype(float), index=s.index)
    dessaz = (s - saz).where(~gap)
    # nível: interpolação linear entre os vizinhos do intervalo
    dessaz = pd.Series(dessaz.to_numpy(), index=np.arange(len(s))).interpolate(limit_area="inside")
    preenchido = dessaz.to_numpy() + saz.to_numpy()
    s[gap] = preenchido[gap]
    return s


# ---------------------------------------------------------------- modelos


def m_ingenuo_sazonal(y: pd.Series, h: int) -> np.ndarray:
    return np.resize(y.to_numpy()[-12:], h)


def m_media_sazonal_2a(y: pd.Series, h: int) -> np.ndarray:
    ult = y.iloc[-24:]
    perfil = ult.groupby(ult.index.month).mean()
    meses = pd.period_range(y.index[-1] + 1, periods=h, freq="M").month
    return perfil.reindex(meses).to_numpy()


def m_holt_winters(y: pd.Series, h: int) -> np.ndarray:
    mod = ExponentialSmoothing(
        y.to_numpy(),
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)
    return mod.forecast(h)


def m_sarima(y: pd.Series, h: int) -> np.ndarray:
    mod = SARIMAX(y.to_numpy(), order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend="n").fit(disp=False)
    return mod.forecast(h)


def m_theta(y: pd.Series, h: int) -> np.ndarray:
    serie = pd.Series(y.to_numpy(), index=y.index.to_timestamp())
    serie.index.freq = "MS"
    return ThetaModel(serie, period=12).fit().forecast(h).to_numpy()


def m_regressao(y: pd.Series, h: int, janela: int = 60) -> np.ndarray:
    ult = y.iloc[-janela:]
    t = np.arange(len(ult))
    X = np.column_stack([np.ones_like(t), t, *[(ult.index.month == m).astype(float) for m in range(2, 13)]])
    beta, *_ = np.linalg.lstsq(X, ult.to_numpy(), rcond=None)
    tf = np.arange(len(ult), len(ult) + h)
    meses = pd.period_range(y.index[-1] + 1, periods=h, freq="M").month
    Xf = np.column_stack([np.ones_like(tf), tf, *[(meses == m).astype(float) for m in range(2, 13)]])
    return Xf @ beta


def m_lightgbm(y: pd.Series, h: int) -> np.ndarray:
    """Modelo direto multi-horizonte: um único LightGBM aprende, a partir de várias origens
    dentro do treino, a relação entre o estado na origem e o valor h meses depois."""
    v = y.to_numpy()
    meses = np.array(y.index.month)
    X, alvo = [], []
    for o in range(24, len(v) - 1):
        hs = min(h, len(v) - 1 - o)
        for k in range(1, hs + 1):
            t = o + k
            X.append(
                [
                    k,
                    meses[t],
                    v[t - 12 * int(np.ceil(k / 12))],
                    v[o - 11 : o + 1].mean(),
                    v[o - 23 : o + 1].mean(),
                    v[o],
                ]
            )
            alvo.append(v[t])
    modelo = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=8,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=SEED,
        verbose=-1,
    ).fit(np.array(X), np.array(alvo))
    o = len(v) - 1
    futuros = pd.period_range(y.index[-1] + 1, periods=h, freq="M").month
    Xf = []
    for k in range(1, h + 1):
        ref = v[o + k - 12 * int(np.ceil(k / 12))]  # mesmo mês, no último ano já observado
        Xf.append([k, futuros[k - 1], ref, v[o - 11 : o + 1].mean(), v[o - 23 : o + 1].mean(), v[o]])
    return modelo.predict(np.array(Xf))


MODELOS = {
    "ingenuo_sazonal": m_ingenuo_sazonal,
    "media_sazonal_2a": m_media_sazonal_2a,
    "holt_winters": m_holt_winters,
    "sarima": m_sarima,
    "theta": m_theta,
    "regressao": m_regressao,
    "lightgbm": m_lightgbm,
}


def prever(nome: str, y: pd.Series, h: int, taxa: bool) -> np.ndarray:
    p = np.asarray(MODELOS[nome](y, h), dtype=float)
    return np.clip(p, 0, 1) if taxa else np.clip(p, 0, None)


# ---------------------------------------------------------------- validação e projeção


def validar(y_bruto: pd.Series, taxa: bool) -> pd.DataFrame:
    linhas = []
    for corte in CORTES:
        treino = preencher_pandemia(y_bruto[y_bruto.index <= corte])
        futuro = y_bruto[(y_bruto.index > corte)].iloc[:HORIZONTE_VALIDACAO]
        futuro = futuro[~((futuro.index >= PANDEMIA[0]) & (futuro.index <= PANDEMIA[1]))]
        if len(futuro) < HORIZONTE_VALIDACAO:
            continue
        for nome in MODELOS:
            try:
                p = prever(nome, treino, HORIZONTE_VALIDACAO, taxa)
            except Exception as exc:  # um modelo que não converge perde aquele corte
                print(f"  [{nome}] falhou no corte {corte}: {exc}")
                continue
            for h, (mes, real) in enumerate(futuro.items(), start=1):
                linhas.append(
                    {
                        "corte": str(corte),
                        "modelo": nome,
                        "h": h,
                        "mes": str(mes),
                        "real": real,
                        "previsto": p[h - 1],
                        "erro": p[h - 1] - real,
                    }
                )
    return pd.DataFrame(linhas)


def main() -> None:
    serie = pd.read_parquet(PROCESSED_DIR / "serie_mensal.parquet")
    serie["mes"] = pd.PeriodIndex(serie["mes"], freq="M")
    SAIDA.mkdir(parents=True, exist_ok=True)

    validacoes, projecoes, resumo = [], [], []
    for escopo, metrica, taxa in SERIES:
        y = serie[serie["escopo"] == escopo].set_index("mes")[metrica].astype(float).sort_index()
        y = y[y.index >= INICIO_SERIE]
        assert len(y) == len(pd.period_range(y.index.min(), y.index.max(), freq="M")), "série com meses faltando"
        print(f"\n== {escopo} · {metrica} ({y.index.min()} a {y.index.max()})")
        val = validar(y, taxa).assign(escopo=escopo, metrica=metrica)
        val["erro_abs"] = val["erro"].abs()
        placar = val.groupby("modelo").agg(mae=("erro_abs", "mean"), vies=("erro", "mean"), cortes=("corte", "nunique"))
        if not taxa:
            soma_real = val.groupby("modelo")["real"].apply(lambda r: r.abs().sum())
            placar["wape"] = val.groupby("modelo")["erro_abs"].sum() / soma_real
        placar = placar.sort_values("mae")
        print(placar.round(4).to_string())
        vencedor = placar.index[0]

        # faixa empírica: P80 do erro absoluto do vencedor por bloco de 6 meses de horizonte
        v = val[val["modelo"] == vencedor].assign(bloco=lambda d: (d["h"] - 1) // 6)
        p80 = v.groupby("bloco")["erro_abs"].quantile(0.8)

        y_cheio = preencher_pandemia(y)
        h = (FIM_PROJECAO - y.index[-1]).n
        central = prever(vencedor, y_cheio, h, taxa)
        meses = pd.period_range(y.index[-1] + 1, periods=h, freq="M")
        largura = np.array([p80.get(min((k - 1) // 6, p80.index.max()), p80.iloc[-1]) for k in range(1, h + 1)])
        inf, sup = central - largura, central + largura
        if taxa:
            inf, sup = np.clip(inf, 0, 1), np.clip(sup, 0, 1)
        else:
            inf = np.clip(inf, 0, None)
        proj = pd.DataFrame(
            {
                "escopo": escopo,
                "metrica": metrica,
                "mes": meses.astype(str),
                "central": central,
                "inferior": inf,
                "superior": sup,
                "modelo": vencedor,
            }
        )
        projecoes.append(proj)
        validacoes.append(val)

        anos = proj.assign(ano=proj["mes"].str[:4]).groupby("ano")[["central", "inferior", "superior"]].mean()
        hist = y[(y.index >= pd.Period("2025-01", freq="M")) & (y.index <= pd.Period("2025-12", freq="M"))].mean()
        resumo.append(
            {
                "escopo": escopo,
                "metrica": metrica,
                "modelo": vencedor,
                "modelo_nome": NOMES[vencedor],
                "mae_validacao": float(placar.loc[vencedor, "mae"]),
                "mae_ingenuo": float(placar.loc["ingenuo_sazonal", "mae"]),
                "ranking": [
                    {"modelo": NOMES[m], "mae": float(r["mae"]), "vies": float(r["vies"])} for m, r in placar.iterrows()
                ],
                "media_2025": float(hist),
                "projecao": {
                    a: {k: float(r[k]) for k in ["central", "inferior", "superior"]} for a, r in anos.iterrows()
                },
            }
        )

    pd.concat(validacoes).to_csv(SAIDA / "validacao.csv", index=False)
    todas = pd.concat(projecoes)
    todas.to_csv(SAIDA / "projecao.csv", index=False)
    (SAIDA / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    exportar_dashboard(serie, todas, resumo)
    print("\nResumo 2027/2028 (média mensal):")
    for r in resumo:
        p = r["projecao"]
        print(
            f"  {r['escopo']:7} {r['metrica']:18} {r['modelo_nome']:36} 2025={r['media_2025']:.4f}  "
            f"2027={p['2027']['central']:.4f} [{p['2027']['inferior']:.4f}-{p['2027']['superior']:.4f}]  "
            f"2028={p['2028']['central']:.4f}"
        )


ROTULOS = {
    ("Brasil", "taxa_atraso"): "Taxa de atraso · Brasil",
    ("Brasil", "taxa_cancelamento"): "Taxa de cancelamento · Brasil",
    ("Brasil", "voos_programados"): "Voos domésticos programados por mês",
    ("Gol", "taxa_atraso"): "Taxa de atraso · Gol",
    ("LATAM", "taxa_atraso"): "Taxa de atraso · LATAM",
    ("Azul", "taxa_atraso"): "Taxa de atraso · Azul",
}


def exportar_dashboard(serie: pd.DataFrame, projecao: pd.DataFrame, resumo: list[dict]) -> None:
    """Tabelas pequenas para a aba de projeção do dashboard (claude.ai e site)."""
    destino = REPORTS_DIR / "dashboard"
    destino.mkdir(parents=True, exist_ok=True)
    linhas = []
    for escopo, metrica, _ in SERIES:
        chave = f"{escopo}|{metrica}"
        y = serie[(serie["escopo"] == escopo) & (serie["mes"] >= INICIO_SERIE)].set_index("mes")[metrica].sort_index()
        pandemia = (y.index >= PANDEMIA[0]) & (y.index <= PANDEMIA[1])
        for mes, v, pan in zip(y.index.astype(str), y.to_numpy(), pandemia):
            linhas.append(
                {
                    "chave": f"{chave}|{mes}",
                    "serie": chave,
                    "mes": mes,
                    "tipo": "pandemia" if pan else "observado",
                    "valor": None if pd.isna(v) else round(float(v), 5),
                    "inferior": None,
                    "superior": None,
                }
            )
        p = projecao[(projecao["escopo"] == escopo) & (projecao["metrica"] == metrica)]
        for r in p.itertuples():
            linhas.append(
                {
                    "chave": f"{chave}|{r.mes}",
                    "serie": chave,
                    "mes": r.mes,
                    "tipo": "projecao",
                    "valor": round(float(r.central), 5),
                    "inferior": round(float(r.inferior), 5),
                    "superior": round(float(r.superior), 5),
                }
            )
    (destino / "projecao_serie.json").write_text(
        json.dumps(linhas, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )

    series, ranking = [], []
    for r in resumo:
        chave = f"{r['escopo']}|{r['metrica']}"
        p27, p28 = r["projecao"]["2027"], r["projecao"]["2028"]
        series.append(
            {
                "serie": chave,
                "rotulo": ROTULOS[(r["escopo"], r["metrica"])],
                "taxa": r["metrica"] != "voos_programados",
                "modelo": r["modelo_nome"],
                "media_2025": round(r["media_2025"], 5),
                "central_2027": round(p27["central"], 5),
                "inferior_2027": round(p27["inferior"], 5),
                "superior_2027": round(p27["superior"], 5),
                "central_2028": round(p28["central"], 5),
                "inferior_2028": round(p28["inferior"], 5),
                "superior_2028": round(p28["superior"], 5),
                "mae_vencedor": round(r["mae_validacao"], 5),
                "mae_ingenuo": round(r["mae_ingenuo"], 5),
                "ganho_vs_ingenuo": round(1 - r["mae_validacao"] / r["mae_ingenuo"], 4) if r["mae_ingenuo"] else 0,
            }
        )
        for pos, m in enumerate(r["ranking"], start=1):
            ranking.append(
                {
                    "chave": f"{chave}|{m['modelo']}",
                    "serie": chave,
                    "posicao": pos,
                    "modelo": m["modelo"],
                    "mae": round(m["mae"], 5),
                    "vies": round(m["vies"], 5),
                }
            )
    (destino / "projecao_series.json").write_text(
        json.dumps(series, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    (destino / "projecao_modelos.json").write_text(
        json.dumps(ranking, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    print(f"[dashboard] {len(linhas)} pontos, {len(series)} séries, {len(ranking)} linhas de ranking")


def figuras() -> None:
    """Figuras do README: projeção da taxa de atraso no Brasil e comparação dos algoritmos."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from atrasos.relatorio import AZUL, GRADE, LARANJA, MUDO, TINTA_2, salvar

    serie = pd.read_parquet(PROCESSED_DIR / "serie_mensal.parquet")
    proj = pd.read_csv(SAIDA / "projecao.csv")
    resumo = json.loads((SAIDA / "resumo.json").read_text(encoding="utf-8"))

    y = serie[(serie["escopo"] == "Brasil") & (serie["mes"] >= str(INICIO_SERIE))].set_index("mes")["taxa_atraso"]
    x = pd.PeriodIndex(y.index, freq="M").to_timestamp()
    p = proj[(proj["escopo"] == "Brasil") & (proj["metrica"] == "taxa_atraso")]
    xp = pd.PeriodIndex(p["mes"], freq="M").to_timestamp()
    pan = (pd.PeriodIndex(y.index, freq="M") >= PANDEMIA[0]) & (pd.PeriodIndex(y.index, freq="M") <= PANDEMIA[1])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(PANDEMIA[0].to_timestamp(), PANDEMIA[1].to_timestamp(how="end"), color=GRADE, alpha=0.6, lw=0)
    ax.annotate("pandemia (lacuna preenchida)", (PANDEMIA[0].to_timestamp(), 0.265), color=TINTA_2, fontsize=8)
    ax.plot(x, y.where(~pan).to_numpy(), color=AZUL, lw=1.6, label="Observado")
    ax.plot(x, y.where(pan).to_numpy(), color=MUDO, lw=1, ls=":", label="Observado na pandemia (não usado)")
    ax.fill_between(
        xp, p["inferior"], p["superior"], color=LARANJA, alpha=0.18, lw=0, label="Faixa de 80% (erros da validação)"
    )
    ax.plot(xp, p["central"], color=LARANJA, lw=2, ls="--", label=f"Projeção ({p['modelo'].iloc[0].upper()})")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylim(0, 0.3)
    ax.set_title("Taxa de atraso dos voos domésticos (> 15 min): histórico e projeção até 2028")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    salvar(fig, "projecao_atraso.png")

    r = next(r for r in resumo if r["escopo"] == "Brasil" and r["metrica"] == "taxa_atraso")
    rk = pd.DataFrame(r["ranking"]).sort_values("mae")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cores = [AZUL if i == 0 else MUDO for i in range(len(rk))]
    ax.barh(rk["modelo"][::-1], rk["mae"][::-1] * 100, color=cores[::-1], height=0.6)
    for i, v in enumerate(rk["mae"][::-1] * 100):
        ax.text(v + 0.03, i, f"{v:.2f} p.p.".replace(".", ",", 1), va="center", fontsize=8, color=TINTA_2)
    ax.set_xlabel("Erro absoluto médio na validação (pontos percentuais, horizontes 1–24 meses)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Qual algoritmo erra menos? Taxa de atraso · Brasil")
    salvar(fig, "projecao_modelos.png")


if __name__ == "__main__":
    main()
    figuras()

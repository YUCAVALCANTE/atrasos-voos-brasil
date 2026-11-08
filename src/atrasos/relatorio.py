"""Gera as figuras do relatório (reports/figures) a partir dos dados e do modelo treinado."""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from atrasos.atributos import como_categoricas
from atrasos.config import FEATURES_PARQUET, FIGURES_DIR, METRICS_PATH, MODEL_PATH, PROCESSED_DIR, REPORTS_DIR, SEED

AZUL, LARANJA, VERDE_AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, MUDO, GRADE, FUNDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

NOMES_MODELOS = {
    "baseline_historico_voo": ("Linha de base: histórico do voo", MUDO),
    "regressao_logistica": ("Regressão logística", LARANJA),
    "lightgbm": ("LightGBM — planejamento", AZUL),
    "lightgbm_operacional": ("LightGBM — operacional (2h antes)", VERDE_AGUA),
}
EMPRESAS = {
    "GLO": "Gol",
    "TAM": "LATAM",
    "AZU": "Azul",
    "PTB": "Voepass",
    "SID": "Sideral",
    "ACN": "Azul Conecta",
    "TTL": "Total",
}

plt.rcParams.update(
    {
        "figure.facecolor": FUNDO,
        "axes.facecolor": FUNDO,
        "savefig.facecolor": FUNDO,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": GRADE,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "xtick.color": MUDO,
        "ytick.color": MUDO,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def salvar(fig, nome: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / nome, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {nome}")


def eda(voos: pd.DataFrame, todos: pd.DataFrame) -> None:
    # Taxa de atraso por hora programada de partida
    por_hora = voos.groupby(voos["partida_prevista"].dt.hour)["atrasado"].mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(por_hora.index, por_hora.values, color=AZUL, width=0.75)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora programada de partida")
    ax.set_title("O atraso se acumula ao longo do dia")
    ax.grid(axis="x", visible=False)
    salvar(fig, "eda_atraso_por_hora.png")

    # Série mensal: atraso (>15 min) e cancelamento
    mensal = pd.DataFrame(
        {
            "Atraso > 15 min": voos.groupby("mes")["atrasado"].mean(),
            "Cancelamento": todos.groupby("mes")["cancelado"].mean(),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = pd.PeriodIndex(mensal.index, freq="M").to_timestamp()
    for col, cor in zip(mensal.columns, [AZUL, LARANJA]):
        ax.plot(x, mensal[col], color=cor, label=col)
        ax.annotate(
            col,
            (x[-1], mensal[col].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=TINTA_2,
            fontsize=9,
        )
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    ax.set_title("Atraso e cancelamento por mês — voos regulares no Brasil")
    salvar(fig, "eda_serie_mensal.png")

    # Empresas com mais voos
    top = voos["empresa"].value_counts().head(6).index
    emp = voos[voos["empresa"].isin(top)].groupby("empresa")["atrasado"].agg(["mean", "size"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rotulos = [f"{EMPRESAS.get(e, e)} ({n / 1e3:,.0f} mil voos)" for e, n in zip(emp.index, emp["size"])]
    ax.barh(rotulos, emp["mean"], color=AZUL, height=0.6)
    for i, v in enumerate(emp["mean"]):
        ax.text(v + 0.003, i, f"{v:.1%}", va="center", color=TINTA_2, fontsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_title("Taxa de atraso por empresa (maiores volumes)")
    salvar(fig, "eda_empresas.png")


def curvas_modelos(pred: pd.DataFrame, metricas: dict) -> None:
    y = pred["atrasado"].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for chave, (nome, cor) in NOMES_MODELOS.items():
        p = pred[f"p_{chave}"].to_numpy()
        m = metricas["modelos"][chave]["teste"]
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=cor, label=f"{nome} — AUC {m['roc_auc']:.3f}")
        prec, rec, _ = precision_recall_curve(y, p)
        a2.plot(rec, prec, color=cor, label=f"{nome} — AP {m['pr_auc']:.3f}")
    a1.plot([0, 1], [0, 1], color=GRADE, lw=1, ls="--")
    a2.axhline(y.mean(), color=GRADE, lw=1, ls="--")
    a1.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos", title="Curva ROC (teste)")
    a2.set(xlabel="Recall", ylabel="Precisão", title="Precisão × Recall (teste)")
    a1.legend(loc="lower right", fontsize=8)
    a2.legend(loc="upper right", fontsize=8)
    salvar(fig, "modelos_roc_pr.png")


def calibracao(pred: pd.DataFrame) -> None:
    y = pred["atrasado"].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=GRADE, lw=1, ls="--", label="Calibração perfeita")
    for chave in ["regressao_logistica", "lightgbm", "lightgbm_operacional"]:
        nome, cor = NOMES_MODELOS[chave]
        fr, mp = calibration_curve(y, pred[f"p_{chave}"], n_bins=15, strategy="quantile")
        ax.plot(mp, fr, color=cor, marker="o", ms=4, label=nome)
    ax.set(xlabel="Probabilidade prevista", ylabel="Frequência observada de atraso", title="Calibração (teste)")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="upper left", fontsize=8)
    salvar(fig, "calibracao.png")


def lift_por_decil(pred: pd.DataFrame) -> pd.DataFrame:
    dec = pd.qcut(pred["p_lightgbm"].rank(method="first"), 10, labels=range(1, 11))
    tab = pred.groupby(dec, observed=True).agg(
        taxa_real=("atrasado", "mean"),
        prob_media=("p_lightgbm", "mean"),
        atraso_medio_min=("atraso_chegada_min", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(tab.index.astype(int), tab["taxa_real"], color=AZUL, width=0.7)
    ax.axhline(pred["atrasado"].mean(), color=TINTA_2, lw=1, ls="--")
    ax.annotate(
        "média geral",
        (0.6, pred["atrasado"].mean()),
        xytext=(0, 4),
        textcoords="offset points",
        color=TINTA_2,
        fontsize=9,
    )
    for d, v in zip(tab.index.astype(int), tab["taxa_real"]):
        ax.text(d, v + 0.008, f"{v:.0%}", ha="center", color=TINTA_2, fontsize=8)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Decil de risco previsto (1 = menor, 10 = maior)")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.set_title("Atraso real por decil de risco do modelo (teste)")
    salvar(fig, "lift_decis.png")
    return tab


def shap_resumo(pacote: dict) -> None:
    import shap

    df = pd.read_parquet(FEATURES_PARQUET)
    te = df[df["divisao"] == "teste"].sample(5000, random_state=SEED)
    te, _ = como_categoricas(te, pacote["categorias"])
    X = te[pacote["features"]]
    valores = shap.TreeExplainer(pacote["modelo"]).shap_values(X)
    if isinstance(valores, list):
        valores = valores[1]
    plt.figure()
    shap.summary_plot(valores, X, max_display=15, show=False, plot_size=(8, 6))
    fig = plt.gcf()
    fig.axes[0].set_title(
        "O que mais pesa na previsão (SHAP, amostra do teste)", loc="left", fontsize=12, fontweight="semibold"
    )
    salvar(fig, "shap_resumo.png")


def main() -> None:
    voos = pd.read_parquet(PROCESSED_DIR / "voos.parquet", columns=["mes", "empresa", "partida_prevista", "atrasado"])
    todos = pd.read_parquet(PROCESSED_DIR / "voos_todos.parquet", columns=["mes", "cancelado"])
    # Mantém só meses completos do período (voos noturnos de set/26 caem em out/26).
    meses_validos = voos["mes"].value_counts()[lambda s: s > 10_000].index
    voos, todos = voos[voos["mes"].isin(meses_validos)], todos[todos["mes"].isin(meses_validos)]
    eda(voos, todos)

    metricas = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    pred = pd.read_parquet(REPORTS_DIR / "predicoes_teste.parquet")
    curvas_modelos(pred, metricas)
    calibracao(pred)
    tab = lift_por_decil(pred)
    tab.to_csv(REPORTS_DIR / "lift_decis.csv")
    shap_resumo(joblib.load(MODEL_PATH))
    print(tab.round(3))


if __name__ == "__main__":
    main()

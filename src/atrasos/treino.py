"""Treino e avaliação: linhas de base vs. LightGBM, com validação temporal e MLflow.

Protocolo:
1. Treino (2024-04 a 2025-12) -> escolha de hiperparâmetros/iterações na validação (2026-S1).
2. Re-treino em treino+validação com o número de iterações escolhido.
3. Avaliação **uma única vez** no teste (2026-T3), que nunca foi visto.
"""

from __future__ import annotations

import json
import time

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from atrasos import __version__
from atrasos.atributos import (
    CATEGORICAS,
    FEATURES,
    FEATURES_OPERACIONAL,
    NUMERICAS,
    como_categoricas,
    congestionamento_tipico,
    tabelas_historicas,
)
from atrasos.config import (
    FEATURES_PARQUET,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DIR,
    REPORTS_DIR,
    ROOT,
    SEED,
)
from atrasos.metricas import avaliar, melhor_limiar_f1

PARAMS_LGB = {
    "objective": "binary",
    # Escolhidos numa busca pequena na validação (ver README, seção 3): árvores menores e
    # categóricas mais regularizadas generalizam melhor para meses futuros.
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "cat_smooth": 100,
    "cat_l2": 50,
    "max_cat_threshold": 16,
    "max_cat_to_onehot": 8,
    "metric": ["auc", "binary_logloss"],
    "verbose": -1,
    "seed": SEED,
}

# A regressão logística não lida bem com milhares de rotas: fica com as categóricas de
# baixa cardinalidade + todas as numéricas (incluindo as taxas históricas).
CATS_LOGISTICA = ["empresa", "tipo_linha", "codigo_di"]


def logistica():
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=500), CATS_LOGISTICA),
            ("num", StandardScaler(), NUMERICAS),
        ]
    )
    return make_pipeline(pre, LogisticRegression(max_iter=300, C=1.0))


def treinar_lgb(
    treino: pd.DataFrame, valid: pd.DataFrame | None, n_iter: int | None = None, features: list[str] = FEATURES
) -> lgb.Booster:
    dtreino = lgb.Dataset(treino[features], treino["atrasado"], categorical_feature=CATEGORICAS)
    if valid is not None:
        dvalid = lgb.Dataset(valid[features], valid["atrasado"], reference=dtreino)
        return lgb.train(
            PARAMS_LGB,
            dtreino,
            num_boost_round=3000,
            valid_sets=[dvalid],
            valid_names=["validacao"],
            callbacks=[lgb.early_stopping(100, first_metric_only=True), lgb.log_evaluation(200)],
        )
    return lgb.train(PARAMS_LGB, dtreino, num_boost_round=n_iter)


def amostra(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=SEED)


def main() -> None:
    # Para ver os experimentos: mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("atrasos-voos-brasil")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PARQUET)
    df = df[df["divisao"] != "historico"].reset_index(drop=True)
    df, categorias = como_categoricas(df)
    tr, va, te = (df[df["divisao"] == d] for d in ["treino", "validacao", "teste"])
    print(f"treino={len(tr):,}  validacao={len(va):,}  teste={len(te):,}")

    resultados: dict[str, dict] = {}
    predicoes_teste = te[
        ["mes", "empresa", "origem", "destino", "hora_partida", "atrasado", "atraso_chegada_min"]
    ].copy()

    # 1) Linha de base ingênua: taxa histórica (suavizada) do próprio número de voo.
    resultados["baseline_historico_voo"] = {
        "validacao": avaliar(va["atrasado"], va["hist_voo_taxa"]),
        "teste": avaliar(te["atrasado"], te["hist_voo_taxa"]),
    }
    predicoes_teste["p_baseline_historico_voo"] = te["hist_voo_taxa"].to_numpy()

    # 2) Regressão logística (treinada numa amostra: ela satura bem antes de milhões de linhas).
    t0 = time.time()
    a_tr = amostra(tr, 500_000)
    lr = logistica().fit(a_tr, a_tr["atrasado"])
    p_va_lr = lr.predict_proba(va)[:, 1]
    a_trva = amostra(pd.concat([tr, va]), 500_000)
    lr_final = logistica().fit(a_trva, a_trva["atrasado"])
    p_te_lr = lr_final.predict_proba(te)[:, 1]
    resultados["regressao_logistica"] = {
        "validacao": avaliar(va["atrasado"], p_va_lr),
        "teste": avaliar(te["atrasado"], p_te_lr),
        "segundos": time.time() - t0,
    }
    predicoes_teste["p_regressao_logistica"] = p_te_lr

    # 3) LightGBM: early stopping na validação -> re-treino em treino+validação.
    with mlflow.start_run(run_name="lightgbm"):
        t0 = time.time()
        booster = treinar_lgb(tr, va)
        melhor_iter = booster.best_iteration
        p_va = booster.predict(va[FEATURES], num_iteration=melhor_iter)
        limiar = melhor_limiar_f1(va["atrasado"].to_numpy(), p_va)

        n_final = int(melhor_iter * 1.1)  # ~10% mais dados -> um pouco mais de árvores
        final = treinar_lgb(pd.concat([tr, va]), None, n_iter=n_final)
        p_te = final.predict(te[FEATURES])
        resultados["lightgbm"] = {
            "validacao": avaliar(va["atrasado"], p_va, limiar),
            "teste": avaliar(te["atrasado"], p_te, limiar),
            "melhor_iteracao": melhor_iter,
            "iteracoes_modelo_final": n_final,
            "segundos": time.time() - t0,
        }
        predicoes_teste["p_lightgbm"] = p_te

        mlflow.log_params({**{k: v for k, v in PARAMS_LGB.items() if k != "metric"}, "n_iter_final": n_final})
        for div in ["validacao", "teste"]:
            mlflow.log_metrics({f"{div}_{k}": v for k, v in resultados["lightgbm"][div].items()})

        importancia = pd.DataFrame({"atributo": FEATURES, "ganho": final.feature_importance("gain")}).sort_values(
            "ganho", ascending=False
        )

        # Artefato servido pela API: modelo + tudo que é preciso para montar atributos de um voo novo.
        voos_hist = pd.read_parquet(
            FEATURES_PARQUET,
            columns=["mes_num", "atrasado", "empresa", "numero_voo", "origem", "destino", "hora_partida"],
        )
        tabelas = tabelas_historicas(voos_hist)
        mes_api = int(voos_hist["mes_num"].max()) + 1  # janela = últimos 3 meses observados
        tabelas = {k: v[v["mes_num"] == mes_api].reset_index(drop=True) for k, v in tabelas.items()}
        malha = pd.read_parquet(
            PROCESSED_DIR / "voos_todos.parquet", columns=["origem", "destino", "partida_prevista", "chegada_prevista"]
        )
        ultimos = malha[malha["partida_prevista"] >= malha["partida_prevista"].max() - pd.Timedelta(days=90)].dropna()
        pacote = {
            "versao": __version__,
            "modelo": final,
            "features": FEATURES,
            "categorias": categorias,
            "tabelas_historicas": tabelas,
            "mes_num_historico": mes_api,
            "congestionamento": congestionamento_tipico(ultimos),
            "limiar": limiar,
            "periodo_treino": [str(tr["mes"].min()), str(va["mes"].max())],
        }
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pacote, MODEL_PATH, compress=3)

    # 4) Cenário operacional: mesmo modelo + o que está acontecendo na operação 2h antes da partida.
    with mlflow.start_run(run_name="lightgbm_operacional"):
        t0 = time.time()
        b_op = treinar_lgb(tr, va, features=FEATURES_OPERACIONAL)
        p_va_op = b_op.predict(va[FEATURES_OPERACIONAL], num_iteration=b_op.best_iteration)
        limiar_op = melhor_limiar_f1(va["atrasado"].to_numpy(), p_va_op)
        n_op = int(b_op.best_iteration * 1.1)
        final_op = treinar_lgb(pd.concat([tr, va]), None, n_iter=n_op, features=FEATURES_OPERACIONAL)
        p_te_op = final_op.predict(te[FEATURES_OPERACIONAL])
        resultados["lightgbm_operacional"] = {
            "validacao": avaliar(va["atrasado"], p_va_op, limiar_op),
            "teste": avaliar(te["atrasado"], p_te_op, limiar_op),
            "melhor_iteracao": b_op.best_iteration,
            "iteracoes_modelo_final": n_op,
            "segundos": time.time() - t0,
        }
        predicoes_teste["p_lightgbm_operacional"] = p_te_op
        mlflow.log_params({"cenario": "operacional", "n_iter_final": n_op})
        for div in ["validacao", "teste"]:
            mlflow.log_metrics({f"{div}_{k}": v for k, v in resultados["lightgbm_operacional"][div].items()})
        importancia_op = pd.DataFrame(
            {"atributo": FEATURES_OPERACIONAL, "ganho": final_op.feature_importance("gain")}
        ).sort_values("ganho", ascending=False)
        importancia_op.to_csv(REPORTS_DIR / "importancia_atributos_operacional.csv", index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predicoes_teste.to_parquet(REPORTS_DIR / "predicoes_teste.parquet", index=False)
    importancia.to_csv(REPORTS_DIR / "importancia_atributos.csv", index=False)
    resumo = {
        "dados": {
            "treino": [str(tr["mes"].min()), str(tr["mes"].max()), len(tr)],
            "validacao": [str(va["mes"].min()), str(va["mes"].max()), len(va)],
            "teste": [str(te["mes"].min()), str(te["mes"].max()), len(te)],
        },
        "modelos": resultados,
    }
    METRICS_PATH.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    tabela = pd.DataFrame({m: r["teste"] for m, r in resultados.items()}).T
    print("\nTESTE (nunca visto):")
    print(tabela[["roc_auc", "pr_auc", "brier", "precisao_top10", "lift_top10"]].round(4))
    print(f"\nlimiar F1 (escolhido na validação) = {limiar:.2f}")
    print(f"tamanho do modelo: {MODEL_PATH.stat().st_size / 1e6:.1f} MB")
    np.set_printoptions(suppress=True)


if __name__ == "__main__":
    main()

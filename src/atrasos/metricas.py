"""Métricas de avaliação — técnicas e de operação."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def avaliar(y: np.ndarray, p: np.ndarray, limiar: float | None = None, top_frac: float = 0.10) -> dict[str, float]:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = y.mean()
    # Precisão no top-k: "se a operação olhar só os 10% de voos de maior risco, quantos atrasam?"
    k = max(1, int(len(p) * top_frac))
    top = np.argsort(-p)[:k]
    prec_top = y[top].mean()
    out = {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "taxa_base": base,
        f"precisao_top{int(top_frac * 100)}": prec_top,
        f"lift_top{int(top_frac * 100)}": prec_top / base,
        f"recall_top{int(top_frac * 100)}": y[top].sum() / max(1, y.sum()),
    }
    if limiar is not None:
        pred = (p >= limiar).astype(int)
        out.update(
            limiar=limiar,
            precisao=precision_score(y, pred, zero_division=0),
            recall=recall_score(y, pred, zero_division=0),
            f1=f1_score(y, pred, zero_division=0),
        )
    return {k: float(v) for k, v in out.items()}


def melhor_limiar_f1(y: np.ndarray, p: np.ndarray) -> float:
    grade = np.linspace(0.05, 0.8, 76)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grade]
    return float(grade[int(np.argmax(f1s))])

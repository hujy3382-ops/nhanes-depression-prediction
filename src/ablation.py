"""hs-CRP ablation and bootstrap confidence intervals."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.train import build_preprocessor, fit_all


def bootstrap_ci(y_true, y_prob, n_boot=1000, seed=42, alpha=0.05):
    """AUROC 点估计 + bootstrap 百分位置信区间。"""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():   # 只有一个类别, AUROC 无定义
            continue
        stats.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return roc_auc_score(y_true, y_prob), lo, hi


def paired_bootstrap_diff(y_true, prob_a, prob_b, n_boot=1000, seed=42, alpha=0.05):
    """AUROC(A) - AUROC(B) 的配对 bootstrap 置信区间 (两模型用同一批重抽样)。"""
    y_true = np.asarray(y_true)
    prob_a, prob_b = np.asarray(prob_a), np.asarray(prob_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(roc_auc_score(y_true[idx], prob_a[idx])
                     - roc_auc_score(y_true[idx], prob_b[idx]))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    return point, lo, hi


def crp_ablation(models, numeric, categorical, X_train, y_train, X_test, y_test,
                 crp_col="LBXHSCRP", n_boot=1000, seed=42) -> pd.DataFrame:
    """比较 '全部特征' 与 '去掉 hs-CRP' 的模型在测试集上的 AUROC。"""
    numeric_wo = [c for c in numeric if c != crp_col]
    cols_wo = numeric_wo + categorical

    full = fit_all(models, build_preprocessor(numeric, categorical), X_train, y_train)
    reduced = fit_all(models, build_preprocessor(numeric_wo, categorical),
                      X_train[cols_wo], y_train)

    rows = []
    for name in models:
        p_full = full[name].predict_proba(X_test)[:, 1]
        p_red = reduced[name].predict_proba(X_test[cols_wo])[:, 1]
        diff, lo, hi = paired_bootstrap_diff(y_test, p_full, p_red, n_boot, seed)
        rows.append({"model": name,
                     "auroc_with_crp": roc_auc_score(y_test, p_full),
                     "auroc_without_crp": roc_auc_score(y_test, p_red),
                     "diff": diff, "diff_ci_low": lo, "diff_ci_high": hi})
    return pd.DataFrame(rows)
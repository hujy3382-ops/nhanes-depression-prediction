"""Day 4 — 在测试集上评估，画 ROC / PR / 校准曲线。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # 只保存图片, 不弹窗口; 避免多线程下的 Tk 报错
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, confusion_matrix,
                             roc_auc_score)


def compute_metrics(y_true, y_prob, threshold: float) -> dict:
    """计算 AUROC、AUPRC、敏感度、特异度。

    提示:
      y_pred = (y_prob >= threshold).astype(int)
      tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
      敏感度 = tp / (tp + fn)；特异度 = tn / (tn + fp)
    思考: 患病率约 9% 时，一个"全部预测为阴性"的模型准确率有多高？为什么不报告 accuracy？
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }


def evaluate_all(fitted: dict, X_test, y_test, threshold: float) -> pd.DataFrame:
    """对每个训练好的模型调用 compute_metrics，汇总成一张表。

    提示: y_prob = pipe.predict_proba(X_test)[:, 1]
    """
    rows = []
    for name, pipe in fitted.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        thr = threshold[name] if isinstance(threshold, dict) else threshold
        rows.append({"model": name, "threshold": thr,
                     **compute_metrics(y_test, y_prob, thr)})
    return pd.DataFrame(rows)


def plot_curves(fitted: dict, X_test, y_test, out_dir: str) -> None:
    """画三张图并保存到 out_dir: roc.png, pr.png, calibration.png，每张图包含所有模型。

    提示:
      fig, ax = plt.subplots(figsize=(6, 6))
      for name, pipe in fitted.items():
          RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
      fig.savefig(Path(out_dir) / "roc.png", dpi=200, bbox_inches="tight")
      plt.close(fig)
    PR 曲线用 PrecisionRecallDisplay，校准曲线用 CalibrationDisplay (n_bins=10)。
    """
    out = Path(out_dir)

    # ROC 曲线
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUROC = 0.5)")
    ax.legend(loc="lower right")
    fig.savefig(out / "roc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # PR 曲线
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipe in fitted.items():
        PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", ls="--", lw=1,
               label=f"Chance (AP = {baseline:.3f})")
    ax.legend(loc="upper right")
    fig.savefig(out / "pr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 校准曲线
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipe in fitted.items():
        CalibrationDisplay.from_estimator(pipe, X_test, y_test, n_bins=10,
                                          name=name, ax=ax)
    fig.savefig(out / "calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


    
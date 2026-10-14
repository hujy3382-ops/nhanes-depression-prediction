"""Day 4 — 在测试集上评估，画 ROC / PR / 校准曲线。"""
from pathlib import Path

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
    raise NotImplementedError("TODO Day 4")


def evaluate_all(fitted: dict, X_test, y_test, threshold: float) -> pd.DataFrame:
    """对每个训练好的模型调用 compute_metrics，汇总成一张表。

    提示: y_prob = pipe.predict_proba(X_test)[:, 1]
    """
    raise NotImplementedError("TODO Day 4")


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
    raise NotImplementedError("TODO Day 4")

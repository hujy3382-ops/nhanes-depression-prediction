"""Day 5 — 用 SHAP 解释模型 (建议只解释 XGBoost)。"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


def shap_summary(pipe, X: pd.DataFrame, out_path: str, max_samples: int = 1000) -> None:
    """为一个训练好的 Pipeline 画 SHAP beeswarm 图并保存。

    难点: SHAP 要解释的是"模型"，而模型看到的是预处理之后的数据。
    提示:
      prep  = pipe.named_steps["prep"]
      model = pipe.named_steps["model"]
      X_t   = prep.transform(X.sample(min(max_samples, len(X)), random_state=0))
      names = prep.get_feature_names_out()
      explainer = shap.TreeExplainer(model)
      shap_values = explainer.shap_values(X_t)
      shap.summary_plot(shap_values, X_t, feature_names=names, show=False)
      plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    加分项: 把 "num__LBXHSCRP" 这种名字替换成可读的 "hs-CRP (log)"。
    """
    raise NotImplementedError("TODO Day 5")

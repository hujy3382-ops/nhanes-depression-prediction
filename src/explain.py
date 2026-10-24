"""Day 5 — 用 SHAP 解释模型 (建议只解释 XGBoost)。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
LABELS = {
    "RIDAGEYR": "Age", "INDFMPIR": "Income-to-poverty ratio",
    "BMXBMI": "BMI", "BMXWAIST": "Waist circumference",
    "LBXHSCRP": "hs-CRP (log)", "LBXWBCSI": "White blood cells",
    "LBXNEPCT": "Neutrophils (%)", "LBXLYPCT": "Lymphocytes (%)",
    "LBXHGB": "Haemoglobin", "LBXPLTSI": "Platelets",
    "LBXGH": "HbA1c", "LBDHDD": "HDL cholesterol", "LBXTC": "Total cholesterol",
    "RIAGENDR": "Sex", "RIDRETH3": "Race/ethnicity", "DMDEDUC2": "Education",
}
CODES = {
    "RIAGENDR": {"1": "Male", "2": "Female"},
    "RIDRETH3": {"1": "Mexican American", "2": "Other Hispanic",
                 "3": "NH White", "4": "NH Black", "6": "NH Asian", "7": "Other/Multi"},
    "DMDEDUC2": {"1": "< 9th grade", "2": "9-11th grade", "3": "High school",
                 "4": "Some college", "5": "College graduate",
                 "7": "Refused", "9": "Don't know"},
}


def pretty_name(raw: str) -> str:
    """'num__LBXHSCRP' -> 'hs-CRP (log)'; 'cat__RIAGENDR_2.0' -> 'Sex: Female'"""
    name = raw.split("__", 1)[1]          # 去掉 num__ / cat__ 前缀
    var, _, level = name.partition("_")    # 分类变量会带上 _类别
    label = LABELS.get(var, var)
    if not level:
        return label
    level = level.removesuffix(".0")
    return f"{label}: {CODES.get(var, {}).get(level, level)}"

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
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    X_t = prep.transform(X.sample(min(max_samples, len(X)), random_state=0))
    if hasattr(X_t, "toarray"):        # 若为稀疏矩阵, 转成普通数组
        X_t = X_t.toarray()
    names = [pretty_name(n) for n in prep.get_feature_names_out()]

    if hasattr(model, "coef_"):        # 线性模型 (逻辑回归)
        explainer = shap.LinearExplainer(model, X_t)
    else:                              # 树模型 (XGBoost)
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_t)

    shap.summary_plot(shap_values, X_t, feature_names=names,
                      max_display=15, show=False)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

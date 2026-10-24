"""Day 3–4 — 构建模型，交叉验证比较多个模型。"""
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def split_data(df: pd.DataFrame, features: list[str], target: str,
               test_size: float, seed: int):
    """划分训练集/测试集，返回 X_train, X_test, y_train, y_test。

    提示: train_test_split(..., stratify=y)
    思考: 抑郁只占约 8–9%，为什么必须 stratify？
    """
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=seed)


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """数值变量: 中位数插补 → 标准化；分类变量: 众数插补 → One-hot 编码。

    提示:
      num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())])
      cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))])
      return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])
    """
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, numeric),
                              ("cat", cat_pipe, categorical)])

def get_models(seed: int, pos_weight: float) -> dict:
    """返回 {模型名: 未训练的模型} 字典。

    建议三个:
      "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced")
      "RandomForest":       RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed)
      "XGBoost":            XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          scale_pos_weight=pos_weight, random_state=seed, eval_metric="logloss")
    pos_weight = 阴性人数 / 阳性人数 (在 main.py 里算好传进来)
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 scale_pos_weight=pos_weight, random_state=seed,
                                 eval_metric="logloss"),
    }

from sklearn.base import clone
def make_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """把预处理和模型串成一个 Pipeline: [("prep", ...), ("model", ...)]。

    面试必问: 为什么插补/标准化要放进 Pipeline，而不是在划分数据之前对整张表做？
    """
    return Pipeline([("prep", clone(preprocessor)),
                     ("model", clone(model))])

def cross_validate_models(models: dict, preprocessor: ColumnTransformer,
                          X: pd.DataFrame, y: pd.Series,
                          folds: int, seed: int) -> pd.DataFrame:
    """对每个模型在训练集上做分层 K 折交叉验证。

    返回 DataFrame，每行一个模型，列: model, auroc_mean, auroc_std, auprc_mean, auprc_std

    提示:
      cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
      scores = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "average_precision"])
      scores["test_roc_auc"].mean()
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        pipe = make_pipeline(preprocessor, model)
        scores = cross_validate(pipe, X, y, cv=cv,
                                scoring=["roc_auc", "average_precision"])
        rows.append({
            "model": name,
            "auroc_mean": scores["test_roc_auc"].mean(),
            "auroc_std": scores["test_roc_auc"].std(),
            "auprc_mean": scores["test_average_precision"].mean(),
            "auprc_std": scores["test_average_precision"].std(),
        })
    return pd.DataFrame(rows)


def fit_all(models: dict, preprocessor: ColumnTransformer,
            X: pd.DataFrame, y: pd.Series) -> dict:
    """在完整训练集上训练每个模型，返回 {模型名: 训练好的 Pipeline}。

    提示: 每个模型都要用 sklearn.base.clone(preprocessor)，避免共用同一个对象。
    """
    fitted = {}
    for name, model in models.items():
        pipe = make_pipeline(preprocessor, model)
        pipe.fit(X, y)
        fitted[name] = pipe
    return fitted


def choose_thresholds(models: dict, preprocessor: ColumnTransformer,
                      X: pd.DataFrame, y: pd.Series,
                      folds: int, seed: int) -> dict:
    """用训练集的折外 (out-of-fold) 预测概率，为每个模型选 Youden 最优阈值。

    只用训练集选阈值，测试集完全不参与，避免数据泄漏。
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    thresholds = {}
    for name, model in models.items():
        pipe = make_pipeline(preprocessor, model)
        oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        fpr, tpr, thr = roc_curve(y, oof)
        thresholds[name] = float(thr[np.argmax(tpr - fpr)])
    return thresholds
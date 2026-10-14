"""Day 3–4 — 构建模型，交叉验证比较多个模型。"""
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
    raise NotImplementedError("TODO Day 3")


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """数值变量: 中位数插补 → 标准化；分类变量: 众数插补 → One-hot 编码。

    提示:
      num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())])
      cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))])
      return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])
    """
    raise NotImplementedError("TODO Day 3")


def get_models(seed: int, pos_weight: float) -> dict:
    """返回 {模型名: 未训练的模型} 字典。

    建议三个:
      "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced")
      "RandomForest":       RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed)
      "XGBoost":            XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          scale_pos_weight=pos_weight, random_state=seed, eval_metric="logloss")
    pos_weight = 阴性人数 / 阳性人数 (在 main.py 里算好传进来)
    """
    raise NotImplementedError("TODO Day 4")


def make_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """把预处理和模型串成一个 Pipeline: [("prep", ...), ("model", ...)]。

    面试必问: 为什么插补/标准化要放进 Pipeline，而不是在划分数据之前对整张表做？
    """
    raise NotImplementedError("TODO Day 3")


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
    raise NotImplementedError("TODO Day 4")


def fit_all(models: dict, preprocessor: ColumnTransformer,
            X: pd.DataFrame, y: pd.Series) -> dict:
    """在完整训练集上训练每个模型，返回 {模型名: 训练好的 Pipeline}。

    提示: 每个模型都要用 sklearn.base.clone(preprocessor)，避免共用同一个对象。
    """
    raise NotImplementedError("TODO Day 4")

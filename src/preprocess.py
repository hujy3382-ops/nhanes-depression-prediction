"""Day 2–3 — 读取、合并、清洗数据，生成分析用的表格。

每个函数只做一件事，这样才方便测试 (见 tests/test_preprocess.py)。
"""
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dir


def load_xpt(path: Path) -> pd.DataFrame:
    """读取一个 .xpt 文件。

    提示: pd.read_sas(path, format="xport")
    ⚠️ NHANES 的坑: 读进来后原本是 0 的值会变成 5.397605e-79 这种极小数。
       用 df.round(6) 或把 abs(x) < 1e-10 的值替换成 0 来修正。
    """
    df = pd.read_sas(path, format="xport")
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].mask(df[num_cols].abs() < 1e-10, 0)
    return df

def merge_on_seqn(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """把多张表按受试者编号 SEQN 合并成一张。

    提示: reduce(lambda a, b: a.merge(b, on="SEQN", how="left"), dfs)
    思考: 为什么用 how="left" 而且第一张表是 DEMO？
    """
    return reduce(lambda a, b: a.merge(b, on="SEQN", how="left"), dfs)


def clean_phq9(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """PHQ-9 每题有效取值 0–3；7 = 拒答，9 = 不知道 → 都改成缺失值 NaN。

    要求: 返回新表，不要修改传进来的 df (先 df = df.copy())。
    提示: df[items] = df[items].where(~df[items].isin([7, 9]))
          (where 的意思: 条件为 True 的保留，否则变成 NaN)
    """
    df = df.copy()
    df[items] = df[items].where(~df[items].isin([7, 9]))
    return df


def compute_phq9(df: pd.DataFrame, items: list[str]) -> pd.Series:
    """计算 PHQ-9 总分 (0–27)。9 题中只要有 1 题缺失，总分就记为 NaN。

    提示: df[items].sum(axis=1, min_count=len(items))
    """
    return df[items].sum(axis=1, min_count=len(items))


def make_label(phq9_total: pd.Series, cutoff: int) -> pd.Series:
    """PHQ-9 >= cutoff 记为 1，否则为 0；总分缺失的保持 NaN。

    提示: 先比较得到 True/False，再用 .where(phq9_total.notna()) 保留缺失。
    """
    label = (phq9_total >= cutoff).astype(float)
    return label.where(phq9_total.notna())


def filter_adults(df: pd.DataFrame, min_age: int) -> pd.DataFrame:
    """只保留年龄 RIDAGEYR >= min_age 的人。"""
    return df[df["RIDAGEYR"] >= min_age].copy()


def log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """对右偏变量取 log1p，覆盖原列。返回新表。

    思考 (面试常问): 为什么用 log1p 而不是 log？
    """
    df = df.copy()
    df[cols] = np.log1p(df[cols])
    return df

def run_preprocessing(config: dict) -> pd.DataFrame:
    """把上面的函数按顺序串起来，并保存到 config["processed_path"]。

    建议流程 (每一步 print 剩余人数 → 这就是你的样本筛选流程图):
      1. 读取 config["files"] 中每个文件 → list of DataFrame (DEMO 放第一个)
      2. merge_on_seqn
      3. filter_adults
      4. clean_phq9 → compute_phq9 → make_label，存为 config["target"] 列
      5. 删除结局缺失的行 (dropna(subset=[target]))
      6. log_transform
      7. 只保留 SEQN + 特征 + 结局 这些列
      8. 保存 csv，返回 DataFrame

    注意: 特征缺失值在这里不要处理！放到 train.py 的 Pipeline 里做 (防止数据泄露)。
    """
    raw_dir = Path(config["raw_dir"])
    suffix = config["suffix"]
    target = config["target"]
    items = config["phq9_items"]

    # 1. 读取 (config 里 DEMO 已经排第一)
    dfs = [load_xpt(raw_dir / f"{name}_{suffix}.xpt") for name in config["files"]]

    # 2. 合并
    df = merge_on_seqn(dfs)
    print(f"Merged participants:        {len(df)}")

    # 3. 筛选成年人
    df = filter_adults(df, config["min_age"])
    print(f"Adults (age >= {config['min_age']}):         {len(df)}")

    # 4. PHQ-9 清洗 → 总分 → 结局
    df = clean_phq9(df, items)
    df["phq9_total"] = compute_phq9(df, items)
    df[target] = make_label(df["phq9_total"], config["phq9_cutoff"])

    # 5. 删除结局缺失
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)
    print(f"With complete PHQ-9:        {len(df)} "
          f"(prevalence {df[target].mean():.1%})")

    # 6. log 转换
    df = log_transform(df, config["log_transform"])

    # 7. 只保留需要的列
    features = config["features"]["numeric"] + config["features"]["categorical"]
    df = df[["SEQN"] + features + [target]]

    # 8. 保存
    out_path = Path(config["processed_path"])
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    print(f"Saved {df.shape} to {out_path}")
    return df
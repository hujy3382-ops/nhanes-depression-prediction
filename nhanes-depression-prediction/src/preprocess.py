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
    raise NotImplementedError("TODO Day 2")


def merge_on_seqn(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """把多张表按受试者编号 SEQN 合并成一张。

    提示: reduce(lambda a, b: a.merge(b, on="SEQN", how="left"), dfs)
    思考: 为什么用 how="left" 而且第一张表是 DEMO？
    """
    raise NotImplementedError("TODO Day 2")


def clean_phq9(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """PHQ-9 每题有效取值 0–3；7 = 拒答，9 = 不知道 → 都改成缺失值 NaN。

    要求: 返回新表，不要修改传进来的 df (先 df = df.copy())。
    提示: df[items] = df[items].where(~df[items].isin([7, 9]))
          (where 的意思: 条件为 True 的保留，否则变成 NaN)
    """
    raise NotImplementedError("TODO Day 3")


def compute_phq9(df: pd.DataFrame, items: list[str]) -> pd.Series:
    """计算 PHQ-9 总分 (0–27)。9 题中只要有 1 题缺失，总分就记为 NaN。

    提示: df[items].sum(axis=1, min_count=len(items))
    """
    raise NotImplementedError("TODO Day 3")


def make_label(phq9_total: pd.Series, cutoff: int) -> pd.Series:
    """PHQ-9 >= cutoff 记为 1，否则为 0；总分缺失的保持 NaN。

    提示: 先比较得到 True/False，再用 .where(phq9_total.notna()) 保留缺失。
    """
    raise NotImplementedError("TODO Day 3")


def filter_adults(df: pd.DataFrame, min_age: int) -> pd.DataFrame:
    """只保留年龄 RIDAGEYR >= min_age 的人。"""
    raise NotImplementedError("TODO Day 3")


def log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """对右偏变量取 log1p，覆盖原列。返回新表。

    思考 (面试常问): 为什么用 log1p 而不是 log？
    """
    raise NotImplementedError("TODO Day 3")


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
    raise NotImplementedError("TODO Day 3")

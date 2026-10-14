"""Day 5 — 单元测试。运行:  pytest -v

这些测试已经写好了。函数实现之前它们会失败，实现正确后就会通过
—— 这就是"测试驱动开发"，可以用来检查你写的函数对不对。
"""
import numpy as np
import pandas as pd

from src.preprocess import clean_phq9, compute_phq9, filter_adults, make_label

ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]


def _phq_df(rows):
    return pd.DataFrame(rows, columns=ITEMS)


def test_clean_phq9_replaces_7_and_9_with_nan():
    df = _phq_df([[0, 1, 2, 3, 7, 9, 0, 0, 0]])
    out = clean_phq9(df, ITEMS)
    assert out[ITEMS[4]].isna().all()
    assert out[ITEMS[5]].isna().all()
    assert out[ITEMS[3]].iloc[0] == 3


def test_clean_phq9_does_not_modify_input():
    df = _phq_df([[7] * 9])
    clean_phq9(df, ITEMS)
    assert (df == 7).all().all()


def test_compute_phq9_sums_items():
    df = _phq_df([[1] * 9, [3] * 9, [0] * 9])
    assert compute_phq9(df, ITEMS).tolist() == [9, 27, 0]


def test_compute_phq9_missing_item_gives_nan():
    df = _phq_df([[1, 1, 1, 1, 1, 1, 1, 1, np.nan]])
    assert np.isnan(compute_phq9(df, ITEMS).iloc[0])


def test_make_label_uses_cutoff_and_keeps_nan():
    total = pd.Series([9, 10, 27, np.nan])
    out = make_label(total, cutoff=10)
    assert out.iloc[:3].tolist() == [0, 1, 1]
    assert np.isnan(out.iloc[3])


def test_filter_adults():
    df = pd.DataFrame({"RIDAGEYR": [12, 17, 18, 80]})
    assert filter_adults(df, 18)["RIDAGEYR"].tolist() == [18, 80]

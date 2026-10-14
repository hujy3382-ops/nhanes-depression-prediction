"""通用小工具。这个文件已经写好了，可以作为"完成的函数长什么样"的参考。"""
from pathlib import Path
import random

import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """读取 YAML 配置文件，返回字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """固定随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> Path:
    """如果文件夹不存在就创建它，返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

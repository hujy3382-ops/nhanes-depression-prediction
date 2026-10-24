"""Day 2 — 从 CDC 官网下载 NHANES 原始数据 (.xpt)。"""
from pathlib import Path

import requests

from src.utils import ensure_dir


def build_url(base_url: str, name: str, year: int, suffix: str) -> str:
    """拼出某个数据文件的下载地址。

    例: build_url(base_url, "DEMO", 2017, "J")
        -> "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt"

    提示: base_url 里有 {year} {name} {suffix} 三个占位符，查一下 str.format()。
    """
    return  base_url.format(year=year, name=name, suffix=suffix)


def download_file(url: str, dest: Path) -> None:
    """把 url 下载到本地 dest。如果 dest 已存在就跳过（不重复下载）。

    提示:
      1. dest.exists() 判断文件是否已存在
      2. requests.get(url, timeout=60)
      3. response.raise_for_status()  —— 下载失败时直接报错，而不是悄悄存一个坏文件
      4. dest.write_bytes(response.content)
      5. print 一行日志，说明下载了什么
    """
    if dest.exists():
        print(f"已存在，跳过：{dest.name}")
        return
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"已下载：{dest.name}")


def download_all(config: dict) -> None:
    """遍历 config["files"]，逐个下载到 config["raw_dir"]。

    提示: 用 ensure_dir 建文件夹; 文件名可以用 f"{name}_{suffix}.xpt"。
    """
    raw_dir = ensure_dir(config["raw_dir"])
    for name in config["files"]:
        url = build_url(config["base_url"], name,
                        config["cycle_year"], config["suffix"])
        dest = raw_dir / f"{name}_{config['suffix']}.xpt"
        download_file(url, dest)

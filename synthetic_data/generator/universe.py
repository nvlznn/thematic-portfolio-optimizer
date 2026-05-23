"""股票池：stock_id、industry、market_cap、theme_tag。

對應 ``§7.2 candidates.parquet``。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import InstanceSpec


@dataclass
class Universe:
    """生成器內部 dataclass — 給其他模組（prices、parameters）共用。"""
    stock_ids: list[str]              # ["SYN0001", ...]，順序固定
    industries: list[str]             # 每檔對應 industry tag
    industry_levels: list[str]        # 該實例所有 industry tag（唯一、排序）
    market_caps: np.ndarray           # shape (N,)
    candidates_df: pd.DataFrame       # §7.2 schema


def _industry_levels(shape: str) -> list[str]:
    if shape == "3_balanced":
        return [f"IND_{c}" for c in "ABC"]
    return [f"IND_{c}" for c in "ABCDE"]


def _draw_industries(
    spec: InstanceSpec, rng: np.random.Generator
) -> tuple[list[str], list[str]]:
    levels = _industry_levels(spec.industry_shape)
    n = spec.n_stocks
    if spec.industry_shape == "5_concentrated":
        probs = np.array([0.60, 0.20, 0.10, 0.05, 0.05])
        idx = rng.choice(len(levels), size=n, p=probs)
        industries = [levels[i] for i in idx]
        # 確保每個 industry 至少 1 檔（concentrated 偶爾會 5、4 完全缺）
        for j, lvl in enumerate(levels):
            if lvl not in industries:
                industries[j % n] = lvl
        return industries, levels
    # balanced：round-robin + 隨機洗牌，每 industry 至少 ⌈n/J⌉ 檔
    industries = [levels[i % len(levels)] for i in range(n)]
    rng.shuffle(industries)
    return industries, levels


def build_universe(spec: InstanceSpec, rng: np.random.Generator) -> Universe:
    n = spec.n_stocks
    stock_ids = [f"SYN{i:04d}" for i in range(1, n + 1)]

    industries, levels = _draw_industries(spec, rng)

    # log-normal market_cap：median ≈ 2.6e10 TWD，>> 5e9 門檻
    market_caps = np.exp(rng.normal(loc=24.0, scale=1.0, size=n))

    df = pd.DataFrame({
        "stock_id": stock_ids,
        "name": [f"Synthetic {sid}" for sid in stock_ids],
        "market_cap": market_caps.astype(np.float64),
        "industry": industries,
        "theme_tag": ["synthetic"] * n,
    })

    return Universe(
        stock_ids=stock_ids,
        industries=industries,
        industry_levels=sorted(set(levels)),
        market_caps=market_caps,
        candidates_df=df,
    )

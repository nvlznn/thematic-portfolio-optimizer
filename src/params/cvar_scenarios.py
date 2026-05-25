"""r_{i,s} CVaR 情境矩陣:歷史視窗 + block bootstrap。

* ``historical`` — 過去 ``lookback_days`` 個交易日,每天的橫斷面報酬向量當一個情境(沿用
  Stage 1 原本作法)。
* ``block_bootstrap`` — 從歷史日報酬抽 ``n_blocks`` 段、每段 ``block_size`` 連續交易日,
  接成 ``n_scenarios`` 列。保留時間自相關與橫斷面結構,擴增尾端樣本(開發計畫 §9 要求
  |Ω|≥250 以免低估風險)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_returns(wide: pd.DataFrame, lookback_days: int, return_type: str) -> pd.DataFrame:
    tail = wide.tail(lookback_days + 1)
    ret = np.log(tail / tail.shift(1)) if return_type == "log" else tail.pct_change()
    return ret.dropna(how="all").fillna(0.0)


def historical_scenarios(wide: pd.DataFrame, lookback_days: int, return_type: str) -> pd.DataFrame:
    """每個交易日 = 一個情境。回傳 (S, N) DataFrame。"""
    return _daily_returns(wide, lookback_days, return_type).reset_index(drop=True)


def block_bootstrap_scenarios(
    wide: pd.DataFrame,
    *,
    lookback_days: int,
    return_type: str,
    n_scenarios: int = 250,
    block_size: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """從歷史日報酬做 moving-block bootstrap,湊出 ``n_scenarios`` 個情境。"""
    base = _daily_returns(wide, lookback_days, return_type).to_numpy()
    n_days = base.shape[0]
    if n_days == 0:
        return pd.DataFrame(columns=wide.columns)
    block_size = max(1, min(block_size, n_days))
    n_blocks = int(np.ceil(n_scenarios / block_size))
    rng = np.random.default_rng(seed)
    max_start = n_days - block_size
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    rows = np.concatenate([base[s : s + block_size] for s in starts], axis=0)[:n_scenarios]
    return pd.DataFrame(rows, columns=wide.columns)


def build_scenarios(
    wide: pd.DataFrame,
    *,
    method: str = "historical",
    lookback_days: int = 240,
    return_type: str = "log",
    n_scenarios: int = 250,
    block_size: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    if method == "historical":
        return historical_scenarios(wide, lookback_days, return_type)
    if method == "block_bootstrap":
        return block_bootstrap_scenarios(
            wide,
            lookback_days=lookback_days,
            return_type=return_type,
            n_scenarios=n_scenarios,
            block_size=block_size,
            seed=seed,
        )
    raise ValueError(f"未知的情境抽樣方式:{method}")

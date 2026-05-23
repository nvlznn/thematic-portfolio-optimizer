"""CVaR 用情境矩陣 (§7.4 scenarios.parquet)。

兩種策略：
1. ``from_prices``：用最後 lookback_days 個交易日的 log-return（與 Stage 1
   ``_build_scenarios`` 一致）。
2. ``synthetic``：直接從 F4 分佈再抽 |Ω| 個情境。

預設用 (1) — 因為 prices 已經涵蓋 F4/F5 的所有結構，scenarios 直接 mirror。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def scenarios_from_prices(
    prices_long: pd.DataFrame,
    stock_ids: list[str],
    n_scenarios: int,
) -> pd.DataFrame:
    """從長表 prices 計算 log-return，取最後 n_scenarios 個交易日。

    輸出 shape (n_scenarios, n_stocks)，index=scenario_id（int），columns=stock_id。
    """
    wide = (
        prices_long.pivot(index="date", columns="stock_id", values="close")
        .sort_index()
        .reindex(columns=stock_ids)
    )
    tail = wide.tail(n_scenarios + 1)
    log_ret = np.log(tail / tail.shift(1)).dropna(how="all").fillna(0.0)
    log_ret = log_ret.tail(n_scenarios).reset_index(drop=True)
    log_ret.index.name = "scenario_id"
    return log_ret

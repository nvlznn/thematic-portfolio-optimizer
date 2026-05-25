"""μ_i 吸引力分數:動能版 + 財務版(blend)。

兩種方法(對應開發計畫 §6「至少 2 種 μ 計算方式」):

* ``momentum``：12-1 動能 — 過去 ``lookback_months`` 個月、跳過最近 ``skip_recent_months``
  個月的對數報酬(避開短期反轉)。
* ``blend``：z-score(動能) × w_mom + z-score(ROE) × w_roe。ROE 取自 params 已接好的
  財務欄(見 src/data/financials.py),先 winsorize 再標準化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import winsorize

TRADING_DAYS_PER_MONTH = 21


def momentum_score(wide: pd.DataFrame, lookback_months: int, skip_recent_months: int) -> pd.Series:
    """以可用天數近似 (L-K)-(K) 月動能;天數不足時自動降階到可用視窗。"""
    n = len(wide)
    skip_days = min(skip_recent_months * TRADING_DAYS_PER_MONTH, max(n - 5, 0))
    lookback_days = min(lookback_months * TRADING_DAYS_PER_MONTH, n - 1)
    end_idx = n - 1 - skip_days
    start_idx = max(end_idx - lookback_days, 0)
    if end_idx <= start_idx:
        return pd.Series(0.0, index=wide.columns, name="mu")
    ret = np.log(wide.iloc[end_idx] / wide.iloc[start_idx])
    return ret.replace([np.inf, -np.inf], np.nan).rename("mu")


def zscore(x: pd.Series) -> pd.Series:
    s = x.std()
    if not np.isfinite(s) or s == 0:
        return x * 0.0
    return (x - x.mean()) / s


def compute_mu(
    wide: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    *,
    method: str = "momentum",
    lookback_months: int = 11,
    skip_recent_months: int = 1,
    blend_weights: dict[str, float] | None = None,
) -> pd.Series:
    """回傳每檔 μ_i(index=wide.columns)。

    ``fundamentals``：以 stock_id 為 index、含 ``roe`` 欄的最新財務快照(blend 才需要)。
    """
    mom = momentum_score(wide, lookback_months, skip_recent_months)
    if method == "momentum":
        return mom.fillna(0.0).rename("mu")
    if method == "blend":
        weights = blend_weights or {"momentum": 0.6, "roe": 0.4}
        if fundamentals is None or "roe" not in fundamentals.columns:
            raise ValueError("blend 需要含 roe 的 fundamentals")
        roe = fundamentals["roe"].reindex(wide.columns)
        z_mom = zscore(mom.fillna(mom.median()))
        z_roe = zscore(winsorize(roe).fillna(roe.median()))
        w_mom = float(weights.get("momentum", 0.5))
        w_roe = float(weights.get("roe", 0.5))
        return (w_mom * z_mom + w_roe * z_roe).fillna(0.0).rename("mu")
    raise ValueError(f"未知的 μ 計算方式:{method}")

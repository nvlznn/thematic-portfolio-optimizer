"""ℓ_i 流動性配置上限:proposal §5.4 的市場衝擊上限 ℓ_i = ρ·ADV / AUM。

ADV = 過去 ``adv_window`` 日成交金額移動平均;單檔可投比例上限 = ρ·ADV/V0,夾到 [0, U]。
缺 ADV(上市未久/停牌)者給 U 的一半作保守上限,避免直接變 0 而無法被選。
"""
from __future__ import annotations

import pandas as pd


def average_daily_value(prices: pd.DataFrame, stocks: list[str], as_of: pd.Timestamp, adv_window: int) -> pd.Series:
    """每檔 ≤ as_of 的最新 ADV(成交金額 adv_window 日移動平均)。"""
    sub = prices[(prices["date"] <= as_of) & (prices["stock_id"].isin(stocks))]
    sub = sub.sort_values(["stock_id", "date"]).copy()
    sub["adv"] = sub.groupby("stock_id")["amount"].transform(
        lambda s: s.rolling(adv_window, min_periods=5).mean()
    )
    return sub.groupby("stock_id")["adv"].last().reindex(stocks)


def liquidity_cap(
    prices: pd.DataFrame,
    stocks: list[str],
    as_of: pd.Timestamp,
    *,
    rho: float,
    adv_window: int,
    V0: float,
    U: float,
) -> pd.Series:
    adv = average_daily_value(prices, stocks, as_of, adv_window)
    cap = (rho * adv / V0).clip(upper=U)
    cap = cap.fillna(U * 0.5).clip(lower=0.0)
    return cap.rename("liquidity_cap")

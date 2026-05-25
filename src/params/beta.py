"""β_i 系統風險:自算滾動 OLS(vs TAIEX)或沿用 TEJ CAPM Beta。

三種來源(``method``):

* ``ols``        — 用個股報酬對大盤(benchmark.parquet 的 Y9999/TAIEX)做 ``window`` 日
  滾動 OLS,β_i = Cov(r_i, r_m) / Var(r_m)。符合開發計畫「自行回歸」的要求。
* ``tej_params`` — 取 params.parquet 已有的 ``beta_3m``(≤ as_of 最新值)。
* ``tej_raw``    — 解析 raw_beta.csv 的 CAPM_Beta(三月)欄。

不論來源,最後都 winsorize/clip 到合理區間並把缺值補成 1.0(市場 β)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import normalize_stock_id

DEFAULT_CLIP = (-1.0, 3.0)


def _finalize(beta: pd.Series, stocks: list[str], clip: tuple[float, float]) -> pd.Series:
    beta = beta.reindex(stocks)
    beta = beta.clip(lower=clip[0], upper=clip[1])
    return beta.fillna(1.0).rename("beta")


def rolling_ols_beta(
    stock_wide: pd.DataFrame,
    market_close: pd.Series,
    *,
    window: int = 120,
    return_type: str = "log",
    min_obs: int = 30,
) -> pd.Series:
    """以最後 ``window`` 個交易日的報酬,對每檔做 vs 大盤的 OLS β。

    ``stock_wide``：寬表收盤價(index=date、columns=stock)。``market_close``:大盤收盤(index=date)。
    """
    if return_type == "log":
        stock_ret = np.log(stock_wide / stock_wide.shift(1))
        mkt_ret = np.log(market_close / market_close.shift(1))
    else:
        stock_ret = stock_wide.pct_change()
        mkt_ret = market_close.pct_change()

    # 對齊大盤可用的交易日,取尾端 window 天
    mkt_ret = mkt_ret.reindex(stock_ret.index)
    common = mkt_ret.dropna().index
    tail = common[-window:]
    if len(tail) < min_obs:
        return pd.Series(np.nan, index=stock_wide.columns, name="beta")

    m = mkt_ret.loc[tail].to_numpy()
    var_m = np.var(m)
    if not np.isfinite(var_m) or var_m <= 0:
        return pd.Series(np.nan, index=stock_wide.columns, name="beta")
    m_centered = m - m.mean()

    betas = {}
    R = stock_ret.loc[tail]
    for stock in stock_wide.columns:
        r = R[stock].to_numpy()
        mask = np.isfinite(r)
        if mask.sum() < min_obs:
            betas[stock] = np.nan
            continue
        rm = m_centered[mask]
        cov = np.dot(r[mask] - r[mask].mean(), rm) / mask.sum()
        var = np.dot(rm, rm) / mask.sum()
        betas[stock] = cov / var if var > 0 else np.nan
    return pd.Series(betas, name="beta")


def beta_from_params(params: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """params.parquet 的 beta_3m,每檔取 ≤ as_of 的最新值。"""
    snap = params[params["date"] <= as_of].dropna(subset=["beta_3m"])
    return snap.sort_values("date").groupby("stock_id")["beta_3m"].last().rename("beta")


def beta_from_raw(raw_beta_path: Path, as_of: pd.Timestamp, *, horizon: str = "3m") -> pd.Series:
    """解析 raw_beta.csv(欄序:代碼, 年月日, CAPM_Beta 三月, CAPM_Beta 一年),取 ≤ as_of 最新。"""
    for encoding, sep in [("utf-16", "\t"), ("cp950", ","), ("utf-8", ",")]:
        try:
            raw = pd.read_csv(raw_beta_path, encoding=encoding, sep=sep, quoting=3)
        except Exception:  # noqa: BLE001
            continue
        if raw.shape[1] >= 4:
            break
    raw = raw.iloc[:, :4]
    raw.columns = ["code_name", "date", "beta_3m", "beta_1y"]
    raw["stock_id"] = normalize_stock_id(raw["code_name"])
    raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d", errors="coerce")
    col = "beta_3m" if horizon == "3m" else "beta_1y"
    raw[col] = pd.to_numeric(raw[col], errors="coerce")
    snap = raw[raw["date"] <= as_of].dropna(subset=[col])
    return snap.sort_values("date").groupby("stock_id")[col].last().rename("beta")


def compute_beta(
    stocks: list[str],
    *,
    method: str = "ols",
    stock_wide: pd.DataFrame | None = None,
    market_close: pd.Series | None = None,
    params: pd.DataFrame | None = None,
    raw_beta_path: Path | None = None,
    as_of: pd.Timestamp | None = None,
    window: int = 120,
    return_type: str = "log",
    clip: tuple[float, float] = DEFAULT_CLIP,
) -> pd.Series:
    """依 ``method`` 估 β,缺值補 1.0、極端值 clip。"""
    if method == "ols":
        if stock_wide is None or market_close is None:
            raise ValueError("ols 需要 stock_wide 與 market_close")
        beta = rolling_ols_beta(stock_wide, market_close, window=window, return_type=return_type)
    elif method == "tej_params":
        if params is None or as_of is None:
            raise ValueError("tej_params 需要 params 與 as_of")
        beta = beta_from_params(params, as_of)
    elif method == "tej_raw":
        if raw_beta_path is None or as_of is None:
            raise ValueError("tej_raw 需要 raw_beta_path 與 as_of")
        beta = beta_from_raw(raw_beta_path, as_of)
    else:
        raise ValueError(f"未知的 beta 來源:{method}")
    return _finalize(beta, stocks, clip)

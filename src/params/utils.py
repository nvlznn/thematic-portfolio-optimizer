"""參數估計共用工具:日期還原、讀處理檔、寬表樞紐。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def decode_dates(series: pd.Series) -> pd.Series:
    """clean.py 把 YYYYMMDD 直接餵 to_datetime 會被當成 ns since epoch(變 1970 年),
    在此偵測並還原成真實日期;若本來就是正常日期則原樣回傳。"""
    as_int = series.astype("int64")
    if (as_int >= 10**7).all() and (as_int < 10**9).all():
        return pd.to_datetime(as_int.astype(str), format="%Y%m%d")
    return pd.to_datetime(series)


def normalize_stock_id(series: pd.Series) -> pd.Series:
    """把可能被污染成「代碼 名稱」的 stock_id 砍成純代碼。"""
    return series.astype(str).str.strip().str.split(r"\s+").str[0]


def load_processed(processed_dir: str | Path, name: str) -> pd.DataFrame:
    """讀 data/processed/<name>.parquet,正規化 stock_id 並還原 date。"""
    df = pd.read_parquet(Path(processed_dir) / f"{name}.parquet")
    if "stock_id" in df.columns:
        df["stock_id"] = normalize_stock_id(df["stock_id"])
    if "date" in df.columns:
        df["date"] = decode_dates(df["date"])
    return df


def pivot_close(prices: pd.DataFrame, stocks: list[str], as_of: pd.Timestamp) -> pd.DataFrame:
    """long prices → wide 收盤價表(index=date ≤ as_of、columns=stocks)。"""
    sub = prices[(prices["date"] <= as_of) & (prices["stock_id"].isin(stocks))]
    wide = sub.pivot_table(index="date", columns="stock_id", values="close").sort_index()
    return wide.reindex(columns=stocks)


def resolve_as_of(prices: pd.DataFrame, override: str | None) -> pd.Timestamp:
    """決定再平衡日:None → prices 最新交易日;否則對齊到 ≤ override 的最大交易日。"""
    if override is None:
        return prices["date"].max()
    ts = pd.Timestamp(override)
    valid = prices.loc[prices["date"] <= ts, "date"]
    if valid.empty:
        raise ValueError(f"找不到 ≤ {override} 的交易日")
    return valid.max()


def winsorize(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """把極端值夾到 [lower_q, upper_q] 分位數之間,避免少數離群值主導 z-score。"""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return s
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return s
    return s.clip(lower=lo, upper=hi)

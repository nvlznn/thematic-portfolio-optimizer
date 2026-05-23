"""Schema validation — §7.1–§7.4 欄位 / dtype / index 一致性。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# 容許 datetime64[us] 與 [ns]（pyarrow 預設寫 us，_decode_dates 都能吃）
_DATETIME_KINDS = ("datetime64[us]", "datetime64[ns]")


def _check(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)


def validate_prices(df: pd.DataFrame) -> list[str]:
    errs: list[str] = []
    expected_cols = {"date", "stock_id", "close", "volume", "amount"}
    _check(set(df.columns) >= expected_cols, f"prices 缺欄位 {expected_cols - set(df.columns)}", errs)
    _check(str(df["date"].dtype) in _DATETIME_KINDS, f"prices.date dtype={df['date'].dtype}", errs)
    _check(df["close"].dtype == "float64", f"prices.close dtype={df['close'].dtype}", errs)
    _check(df["volume"].dtype == "int64", f"prices.volume dtype={df['volume'].dtype}", errs)
    _check(df["amount"].dtype == "float64", f"prices.amount dtype={df['amount'].dtype}", errs)
    _check(not df.duplicated(subset=["date", "stock_id"]).any(), "prices (date,stock_id) 重複", errs)
    return errs


def validate_candidates(df: pd.DataFrame) -> list[str]:
    errs: list[str] = []
    expected_cols = {"stock_id", "name", "market_cap", "industry", "theme_tag"}
    _check(set(df.columns) >= expected_cols, f"candidates 缺欄位 {expected_cols - set(df.columns)}", errs)
    _check(df["stock_id"].is_unique, "candidates.stock_id 不唯一", errs)
    _check(df["market_cap"].dtype == "float64", f"candidates.market_cap dtype={df['market_cap'].dtype}", errs)
    return errs


def validate_params(df: pd.DataFrame, expected_stocks: set[str]) -> list[str]:
    errs: list[str] = []
    expected_cols = {"date", "stock_id", "close", "market_cap", "beta_3m", "roe", "revenue_growth"}
    _check(set(df.columns) >= expected_cols, f"params 缺欄位 {expected_cols - set(df.columns)}", errs)
    _check(str(df["date"].dtype) in _DATETIME_KINDS, f"params.date dtype={df['date'].dtype}", errs)
    actual = set(df["stock_id"].unique())
    _check(actual == expected_stocks, f"params stock_id 與 candidates 不一致：缺 {expected_stocks - actual}, 多 {actual - expected_stocks}", errs)
    return errs


def validate_scenarios(df: pd.DataFrame, expected_stocks: list[str], n_scenarios: int) -> list[str]:
    errs: list[str] = []
    _check(df.shape == (n_scenarios, len(expected_stocks)), f"scenarios shape={df.shape}, 預期 ({n_scenarios}, {len(expected_stocks)})", errs)
    actual = list(df.columns)
    _check(actual == expected_stocks, "scenarios columns 與 stock_ids 不一致或順序不對", errs)
    return errs


def validate_instance(instance_dir: Path) -> dict:
    errs: list[str] = []
    prices = pd.read_parquet(instance_dir / "prices.parquet")
    cands = pd.read_parquet(instance_dir / "candidates.parquet")
    params = pd.read_parquet(instance_dir / "params.parquet")
    scen = pd.read_parquet(instance_dir / "scenarios.parquet")

    errs += [f"[prices] {e}" for e in validate_prices(prices)]
    errs += [f"[candidates] {e}" for e in validate_candidates(cands)]
    stock_set = set(cands["stock_id"])
    errs += [f"[params] {e}" for e in validate_params(params, stock_set)]
    stock_list = cands["stock_id"].tolist()
    errs += [f"[scenarios] {e}" for e in validate_scenarios(scen, stock_list, n_scenarios=scen.shape[0])]
    return {"errors": errs, "ok": len(errs) == 0}

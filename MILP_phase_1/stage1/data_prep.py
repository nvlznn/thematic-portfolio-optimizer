"""Stage1 資料準備：從 §7.1–§7.3 介面契約檔轉成 MILP 輸入。

從專案根目錄 ``data/processed/`` 讀入 ``prices.parquet``、``candidates.parquet``、
``params.parquet``，產出 Stage1 模型所需的：

    * universe：通過篩選的候選股代碼清單
    * μ_i：投資吸引力分數
    * β_i：Beta
    * ℓ_i：流動性配置上限
    * w0_i：期初權重
    * S_j：產業分群
    * r_{i,s}：CVaR 用報酬情境矩陣（情境 × 股票）

輸入檔由 ``src/data/clean.py`` 產生；其 ``date`` 欄位的數值是把 YYYYMMDD 直接
解讀為 ns since epoch，所以這裡的 :func:`_decode_dates` 會還原成真實日期。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOT_SIZE = 1000  # 台股 1 張 = 1000 股


@dataclass
class Stage1Inputs:
    """Stage1 MILP 的全部輸入。"""

    rebalance_date: pd.Timestamp
    stocks: list[str]                     # universe I，順序固定
    mu: np.ndarray                        # shape (N,)
    beta: np.ndarray                      # shape (N,)
    liquidity_cap: np.ndarray             # shape (N,) — proposal §5.4 的 ρ·ADV/AUM
    price_per_lot: np.ndarray             # shape (N,) — P_i（元/張，含 lot_size）
    L_per_stock: np.ndarray               # shape (N,) — 每檔的最低可達權重 = L_lot·P/V0
    U_per_stock: np.ndarray               # shape (N,) — 每檔的最高可達權重 = U_lot·P/V0
    w0: np.ndarray                        # shape (N,)
    industries: list[str]                 # shape (N,)，每檔對應的產業
    industry_groups: dict[str, list[int]] # 產業 -> 該產業在 stocks 中的 index list
    scenarios: np.ndarray                 # shape (S, N) 情境 r_{i,s}
    diagnostics: dict                     # 過程紀錄


def _decode_dates(series: pd.Series) -> pd.Series:
    """clean.py 把 YYYYMMDD 直接餵給 :func:`pd.to_datetime`，被當成 ns since epoch；
    這裡偵測那種情況並反解回真實日期。"""
    as_int = series.astype("int64")
    # 真實 ns 時間戳 >= 10^17（>=1973 年）；YYYYMMDD 落在 [10^7, 10^8)
    if (as_int >= 10**7).all() and (as_int < 10**9).all():
        return pd.to_datetime(as_int.astype(str), format="%Y%m%d")
    return pd.to_datetime(series)


def load_raw(processed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = Path(processed_dir)
    prices = pd.read_parquet(p / "prices.parquet")
    cands = pd.read_parquet(p / "candidates.parquet")
    params = pd.read_parquet(p / "params.parquet")
    prices["date"] = _decode_dates(prices["date"])
    params["date"] = _decode_dates(params["date"])
    return prices, cands, params


def _resolve_rebalance_date(prices: pd.DataFrame, override: str | None) -> pd.Timestamp:
    if override is None:
        return prices["date"].max()
    ts = pd.Timestamp(override)
    # 對齊到 <= override 的最大交易日
    valid = prices.loc[prices["date"] <= ts, "date"]
    if valid.empty:
        raise ValueError(f"找不到 <= {override} 的交易日")
    return valid.max()


def _build_universe(
    prices: pd.DataFrame,
    cands: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    min_market_cap: float,
    min_history_days: int,
    require_industry: bool,
) -> list[str]:
    history = prices[prices["date"] <= rebalance_date]
    counts = history.groupby("stock_id").size()
    has_history = counts[counts >= min_history_days].index

    pool = cands.copy()
    if require_industry:
        pool = pool[pool["industry"].notna()]
    pool = pool[pool["market_cap"] >= min_market_cap]
    pool = pool[pool["stock_id"].isin(has_history)]
    return sorted(pool["stock_id"].unique().tolist())


def _wide_prices(prices: pd.DataFrame, stocks: list[str], rebalance_date: pd.Timestamp) -> pd.DataFrame:
    sub = prices[(prices["date"] <= rebalance_date) & (prices["stock_id"].isin(stocks))]
    wide = sub.pivot(index="date", columns="stock_id", values="close").sort_index()
    return wide.reindex(columns=stocks)


def _momentum(wide: pd.DataFrame, lookback_months: int, skip_recent_months: int) -> pd.Series:
    """以可用天數近似實現 (L-K)-(K) 月動能。"""
    n = len(wide)
    # 每月約 21 個交易日
    skip_days = min(skip_recent_months * 21, max(n - 5, 0))
    lookback_days = min(lookback_months * 21, n - 1)
    end_idx = n - 1 - skip_days
    start_idx = max(end_idx - lookback_days, 0)
    if end_idx <= start_idx:
        return pd.Series(0.0, index=wide.columns, name="mu")
    p_end = wide.iloc[end_idx]
    p_start = wide.iloc[start_idx]
    ret = np.log(p_end / p_start)
    ret = ret.replace([np.inf, -np.inf], np.nan)
    return ret.rename("mu")


def _z_score(x: pd.Series) -> pd.Series:
    s = x.std()
    if not np.isfinite(s) or s == 0:
        return x * 0.0
    return (x - x.mean()) / s


def _build_mu(
    wide: pd.DataFrame,
    params_latest: pd.DataFrame,
    cfg: dict,
) -> pd.Series:
    method = cfg.get("method", "momentum")
    mom = _momentum(wide, cfg["lookback_months"], cfg["skip_recent_months"])
    if method == "momentum":
        return mom.fillna(0.0)
    if method == "blend":
        roe = params_latest.set_index("stock_id")["roe"].reindex(wide.columns)
        z_mom = _z_score(mom.fillna(mom.median()))
        z_roe = _z_score(roe.fillna(roe.median()))
        wm = cfg["blend_weights"].get("momentum", 0.5)
        wr = cfg["blend_weights"].get("roe", 0.5)
        return (wm * z_mom + wr * z_roe).fillna(0.0).rename("mu")
    raise ValueError(f"未知的 μ 計算方式：{method}")


def _build_beta(params: pd.DataFrame, stocks: list[str], rebalance_date: pd.Timestamp) -> pd.Series:
    snap = params[params["date"] <= rebalance_date]
    latest_beta = (
        snap.dropna(subset=["beta_3m"])
        .sort_values("date")
        .groupby("stock_id")["beta_3m"]
        .last()
        .reindex(stocks)
    )
    # 缺值用 1.0 作市場 beta 代填
    return latest_beta.fillna(1.0).rename("beta")


def _build_liquidity(
    prices: pd.DataFrame,
    stocks: list[str],
    rebalance_date: pd.Timestamp,
    rho: float,
    adv_window: int,
    V0: float,
    U: float,
) -> pd.Series:
    """proposal §5.4 的市場衝擊上限 ℓ_i = ρ·ADV/AUM。

    整數張數可行性改由 :func:`_build_lot_bounds` 透過 U_per_stock = 0 直接讓
    模型把那種股票排除掉，職責更清楚。
    """
    sub = prices[(prices["date"] <= rebalance_date) & (prices["stock_id"].isin(stocks))]
    sub = sub.sort_values(["stock_id", "date"]).copy()
    sub["amount_ma"] = (
        sub.groupby("stock_id")["amount"].transform(lambda s: s.rolling(adv_window, min_periods=5).mean())
    )
    adv = sub.groupby("stock_id")["amount_ma"].last().reindex(stocks)
    cap = (rho * adv / V0).clip(upper=U)
    # ADV 過小者用 U 的一半作下限上限，避免 0 直接讓股票無法被選
    cap = cap.fillna(U * 0.5).clip(lower=0.0)
    return cap.rename("liquidity_cap")


def _build_lot_bounds(
    prices: pd.DataFrame,
    stocks: list[str],
    rebalance_date: pd.Timestamp,
    L_global: float,
    U_global: float,
    V0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """回傳 (price_per_lot, L_per_stock, U_per_stock, lot_infeasible_list)。

    與 Stage 2 的整數張數可行域 **完全一致**：

    * ``L_per_stock_i = ⌈L·V0 / P_lot_i⌉ · P_lot_i / V0``
    * ``U_per_stock_i = ⌊U·V0 / P_lot_i⌋ · P_lot_i / V0``

    這把 Stage 2 的張數鏈接限制 ``L_lot·z ≤ x ≤ U_lot·z`` 投影回 Stage 1 的權重空間，
    讓兩階段在 w 軸上有相同的可行域。當 ``⌊U·V0/P_lot⌋ < 1`` 時（1 張已超過 U·V0），
    L_per_stock 與 U_per_stock 同時為 0，模型自動把該股 z 強制為 0。
    """
    sub = prices[(prices["date"] <= rebalance_date) & (prices["stock_id"].isin(stocks))]
    latest_px = (
        sub.sort_values("date").groupby("stock_id")["close"].last().reindex(stocks)
    )
    P_lot = (latest_px * LOT_SIZE).to_numpy(dtype=float)

    valid = (P_lot > 0) & np.isfinite(P_lot)
    L_lot = np.zeros_like(P_lot, dtype=np.int64)
    U_lot = np.zeros_like(P_lot, dtype=np.int64)
    L_lot[valid] = np.ceil(L_global * V0 / P_lot[valid]).astype(np.int64)
    U_lot[valid] = np.floor(U_global * V0 / P_lot[valid]).astype(np.int64)

    # 1 張都塞不進 U·V0 → 不可選
    infeasible_mask = ~valid | (U_lot < 1)
    infeasible = [s for s, flag in zip(stocks, infeasible_mask) if flag]

    L_per_stock = np.where(infeasible_mask, 0.0, L_lot * P_lot / V0)
    U_per_stock = np.where(infeasible_mask, 0.0, U_lot * P_lot / V0)
    P_lot = np.nan_to_num(P_lot, nan=0.0)

    return P_lot, L_per_stock, U_per_stock, infeasible


def _build_scenarios(
    wide: pd.DataFrame,
    lookback_days: int,
    return_type: str,
) -> pd.DataFrame:
    tail = wide.tail(lookback_days + 1)
    if return_type == "log":
        ret = np.log(tail / tail.shift(1))
    else:
        ret = tail.pct_change()
    ret = ret.dropna(how="all")
    # 個別缺值填 0（停牌 / 上市未久）
    ret = ret.fillna(0.0)
    return ret


def _initial_weights(stocks: list[str], init_map: dict[str, float] | None) -> pd.Series:
    s = pd.Series(0.0, index=stocks, name="w0")
    if not init_map:
        return s
    for k, v in init_map.items():
        if k in s.index:
            s.loc[k] = float(v)
    return s


def build_stage1_inputs(
    processed_dir: str | Path,
    cfg: dict,
) -> Stage1Inputs:
    prices, cands, params = load_raw(processed_dir)

    rebalance_date = _resolve_rebalance_date(prices, cfg["portfolio"].get("rebalance_date"))

    stocks = _build_universe(
        prices,
        cands,
        rebalance_date,
        min_market_cap=float(cfg["universe"]["min_market_cap"]),
        min_history_days=int(cfg["universe"]["min_history_days"]),
        require_industry=bool(cfg["universe"]["require_industry"]),
    )
    if not stocks:
        raise RuntimeError("候選股池為空 — 檢查篩選條件")

    wide = _wide_prices(prices, stocks, rebalance_date)
    params_latest = (
        params[params["date"] <= rebalance_date]
        .sort_values("date")
        .groupby("stock_id")
        .last()
        .reset_index()
    )

    mu = _build_mu(wide, params_latest, cfg["mu"]).reindex(stocks).fillna(0.0)
    beta = _build_beta(params, stocks, rebalance_date)
    V0 = float(cfg["portfolio"]["V0"])
    L_global = float(cfg["stage1"]["L"])
    U_global = float(cfg["stage1"]["U"])

    liq = _build_liquidity(
        prices, stocks, rebalance_date,
        rho=float(cfg["liquidity"]["rho"]),
        adv_window=int(cfg["liquidity"]["adv_window"]),
        V0=V0,
        U=U_global,
    )
    P_lot, L_per_stock, U_per_stock, lot_infeasible = _build_lot_bounds(
        prices, stocks, rebalance_date,
        L_global=L_global,
        U_global=U_global,
        V0=V0,
    )
    w0 = _initial_weights(stocks, cfg["portfolio"].get("initial_weights"))

    industry_map = cands.set_index("stock_id")["industry"].to_dict()
    industries = [industry_map.get(s, "其他") or "其他" for s in stocks]
    groups: dict[str, list[int]] = {}
    for idx, ind in enumerate(industries):
        groups.setdefault(ind, []).append(idx)

    scenarios_df = _build_scenarios(
        wide,
        lookback_days=cfg["scenarios"]["lookback_days"],
        return_type=cfg["scenarios"]["return_type"],
    )
    scenarios = scenarios_df.reindex(columns=stocks).fillna(0.0).to_numpy()

    selectable = U_per_stock > 0
    n_selectable = int(selectable.sum())
    L_sel = L_per_stock[selectable]
    diagnostics = {
        "n_universe": len(stocks),
        "n_selectable": n_selectable,
        "n_industries": len(groups),
        "n_scenarios": int(scenarios.shape[0]),
        "n_lot_infeasible": len(lot_infeasible),
        "lot_infeasible_examples": lot_infeasible[:20],
        "L_per_stock_summary": {
            "global_L": L_global,
            "median": float(np.median(L_sel)) if n_selectable else 0.0,
            "max": float(L_sel.max()) if n_selectable else 0.0,
            "n_above_global": int((L_sel > L_global + 1e-9).sum()),
        },
        "mu_summary": {
            "min": float(mu.min()),
            "max": float(mu.max()),
            "mean": float(mu.mean()),
        },
        "beta_summary": {
            "min": float(beta.min()),
            "max": float(beta.max()),
            "mean": float(beta.mean()),
        },
    }

    return Stage1Inputs(
        rebalance_date=rebalance_date,
        stocks=stocks,
        mu=mu.to_numpy(),
        beta=beta.to_numpy(),
        liquidity_cap=liq.to_numpy(),
        price_per_lot=P_lot,
        L_per_stock=L_per_stock,
        U_per_stock=U_per_stock,
        w0=w0.to_numpy(),
        industries=industries,
        industry_groups=groups,
        scenarios=scenarios,
        diagnostics=diagnostics,
    )

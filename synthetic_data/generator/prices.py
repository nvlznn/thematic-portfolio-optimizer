"""Factor-model GBM 價格模擬。

每日 log-return：
    r_i[t] = drift_i + β_loading_i · r_market[t]
             + λ_industry_i · r_industry[j[i],t]   (F5=block 時)
             + ε_i[t]

F4 控制 r_market 的分佈；F5 控制 β_loading（independent → 0，factor → β_i，
block → β_i + 共用 industry shock）。

對應 ``prices.parquet``（§7.1）：date / stock_id / close / volume / amount。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import InstanceSpec
from .parameters import StockParameters
from .universe import Universe

TRADING_DAYS_PER_YEAR = 252
N_DAYS_DEFAULT = 320  # >= 250 (scenarios lookback) + 50 (margin)
AS_OF = pd.Timestamp("2026-04-30")

ANN_MU_MARKET = 0.06
ANN_SIGMA_MARKET = 0.18
ANN_SIGMA_INDUSTRY = 0.08
T_DF = 5  # 學生 t 自由度


def _business_dates(n_days: int, end: pd.Timestamp = AS_OF) -> pd.DatetimeIndex:
    """產生長度 n_days 的營業日序列，最後一日對齊 end。"""
    return pd.bdate_range(end=end, periods=n_days)


def _draw_market(spec: InstanceSpec, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """市場日 log-return（已年化轉日化）。"""
    daily_mu = ANN_MU_MARKET / TRADING_DAYS_PER_YEAR
    daily_sigma = ANN_SIGMA_MARKET / np.sqrt(TRADING_DAYS_PER_YEAR)
    if spec.return_dist == "gaussian":
        return daily_mu + daily_sigma * rng.standard_normal(n_days)
    if spec.return_dist == "t":
        # 用 t/sqrt(var) 標準化，再乘 σ → 變異數與 Gaussian 一致，只是 tail 較肥
        t = rng.standard_t(df=T_DF, size=n_days)
        t = t / np.sqrt(T_DF / (T_DF - 2))  # var(scaled t) = 1
        return daily_mu + daily_sigma * t
    if spec.return_dist == "bootstrap":
        # 先生 Gaussian 5 年的 path，再 block-resample 5-day blocks
        long_path = daily_mu + daily_sigma * rng.standard_normal(5 * TRADING_DAYS_PER_YEAR)
        block_len = 5
        n_blocks = (n_days + block_len - 1) // block_len
        starts = rng.integers(low=0, high=len(long_path) - block_len, size=n_blocks)
        chunks = [long_path[s : s + block_len] for s in starts]
        return np.concatenate(chunks)[:n_days]
    raise ValueError(f"未知 return_dist: {spec.return_dist}")


def _industry_shocks(
    universe: Universe, n_days: int, rng: np.random.Generator
) -> np.ndarray:
    """shape (n_days, n_stocks)，每檔股票套用其所屬產業的當日 shock。"""
    daily_sigma = ANN_SIGMA_INDUSTRY / np.sqrt(TRADING_DAYS_PER_YEAR)
    n = len(universe.stock_ids)
    inds = universe.industry_levels
    ind_idx_per_stock = np.array([inds.index(g) for g in universe.industries], dtype=int)
    shocks_per_ind = daily_sigma * rng.standard_normal(size=(n_days, len(inds)))
    return shocks_per_ind[:, ind_idx_per_stock]


def simulate_prices(
    spec: InstanceSpec,
    universe: Universe,
    params: StockParameters,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """主入口：回傳長表 prices.parquet 對應的 DataFrame。"""
    n_days = N_DAYS_DEFAULT
    n_stocks = spec.n_stocks
    dates = _business_dates(n_days)

    market_r = _draw_market(spec, n_days, rng)            # shape (T,)
    # 每股每日的個別噪音（年化 σ_idio_i 已存於 params）
    daily_sigma_idio = params.sigma_idio / np.sqrt(TRADING_DAYS_PER_YEAR)
    idio = rng.standard_normal((n_days, n_stocks)) * daily_sigma_idio[np.newaxis, :]

    # β loading 依 F5 切換
    if spec.corr_structure == "independent":
        beta_loading = np.zeros(n_stocks)
    else:
        # factor / block 都用 β_i 當 market loading
        beta_loading = params.beta

    # block 需額外加 industry shock
    if spec.corr_structure == "block":
        ind_shock = _industry_shocks(universe, n_days, rng)  # shape (T, N)
    else:
        ind_shock = 0.0

    # 個別 drift：對 mu_override 加一點點 drift 讓 _build_mu 重算的 momentum 與 mu_override 同向
    # （不要太強，避免價格爆炸；只是讓「正 μ → 正動能」這條對齊變得自然）
    daily_drift = (params.mu_override * 0.05) / TRADING_DAYS_PER_YEAR  # 5% 年化 / σ_μ_override≈1

    # 組裝日 log-return
    log_r = (
        daily_drift[np.newaxis, :]
        + beta_loading[np.newaxis, :] * market_r[:, np.newaxis]
        + ind_shock
        + idio
    )  # shape (T, N)

    # 累積成價格
    log_p0 = np.log(params.P0)
    log_p = log_p0[np.newaxis, :] + np.cumsum(log_r, axis=0)
    close = np.exp(log_p)  # shape (T, N)

    # 成交量/額：amount_t ≈ ADV20 · (1 + ε)，volume = amount / price
    eps = rng.normal(loc=0.0, scale=0.10, size=(n_days, n_stocks))
    amount = params.adv20_amount[np.newaxis, :] * np.clip(1.0 + eps, 0.1, None)
    volume = (amount / close).round().astype(np.int64)

    # 轉長表
    rows = []
    for t_idx, date in enumerate(dates):
        for s_idx, sid in enumerate(universe.stock_ids):
            rows.append((date, sid, float(close[t_idx, s_idx]),
                         int(volume[t_idx, s_idx]), float(amount[t_idx, s_idx])))
    return pd.DataFrame(rows, columns=["date", "stock_id", "close", "volume", "amount"])

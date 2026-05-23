"""β、μ、liquidity、initial state — 合併 PLAN §3.4–3.6 的三個小模組。

對應 ``params.parquet``（§7.3）以及每實例的 ``stage_overrides``（注入 Stage 1/2 config）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .factors import InstanceSpec

LOT_SIZE = 1000  # 對齊 stage1/data_prep.py


@dataclass
class StockParameters:
    beta: np.ndarray              # shape (N,)，target β（也餵進價格模擬器當 loading）
    mu_override: np.ndarray       # shape (N,)，z-scored attractiveness
    roe: np.ndarray
    revenue_growth: np.ndarray
    P0: np.ndarray                # 每股初始價格（元/股）
    adv20_amount: np.ndarray      # 期望 20 日平均成交金額（元）
    sigma_idio: np.ndarray        # 個股 idiosyncratic 年化 σ（log-return）


@dataclass
class InitialState:
    w0_map: dict[str, float]      # Stage 1 initial_weights
    x0_lots_map: dict[str, int]   # Stage 2 initial_holdings


# ---------------- β ---------------- #

def draw_beta(spec: InstanceSpec, rng: np.random.Generator) -> np.ndarray:
    n = spec.n_stocks
    if spec.beta_dist == "tight":
        b = rng.normal(loc=1.0, scale=0.15, size=n)
    else:  # wide
        b = rng.uniform(low=0.3, high=1.8, size=n)
    return np.clip(b, -1.0, 3.0)


# ---------------- μ：經 Cholesky 耦合到 β ---------------- #

def draw_mu(spec: InstanceSpec, beta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rho = float(spec.mu_beta_corr)
    z_beta = (beta - beta.mean()) / (beta.std() + 1e-12)
    z_raw = rng.standard_normal(size=beta.shape)
    z_mu = rho * z_beta + np.sqrt(max(0.0, 1.0 - rho * rho)) * z_raw
    return z_mu  # 已是 z-score（mean≈0, std≈1）


# ---------------- 其他 per-stock 常數 ---------------- #

def draw_financials(spec: InstanceSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """ROE 與 revenue_growth — Stage 1 預設不用，但 §7.3 要求欄位存在。"""
    n = spec.n_stocks
    roe = rng.normal(loc=0.10, scale=0.05, size=n)
    rev = rng.normal(loc=0.05, scale=0.10, size=n)
    return roe, rev


def draw_prices_seed(spec: InstanceSpec, rng: np.random.Generator) -> np.ndarray:
    """每股初始價格 P0_i（元/股）。F8 控制。"""
    n = spec.n_stocks
    if spec.price_dist == "uniform":
        return rng.uniform(low=20.0, high=200.0, size=n)
    # log-normal：median 80, σ=0.8 → 偶爾出現 500+ 的高股價
    return rng.lognormal(mean=np.log(80.0), sigma=0.8, size=n)


def draw_idio_sigma(spec: InstanceSpec, rng: np.random.Generator) -> np.ndarray:
    """個股 idiosyncratic 年化 σ ∈ [0.10, 0.30]。"""
    n = spec.n_stocks
    return rng.uniform(low=0.10, high=0.30, size=n)


# ---------------- 流動性 ---------------- #

def draw_liquidity(
    spec: InstanceSpec,
    P0: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """回傳期望 ADV20（元）。

    對非 stress 實例：``target_liq_cap_i ~ U[2L, U]``，然後反推
    ``ADV20_i = target_liq_cap_i · V0 / ρ``（ρ=0.10 對齊 stage1/config）。

    F10=stress 時，10–20% 股票的 liq_cap 故意低於 L 來觸發 liquidity binding。
    """
    n = spec.n_stocks
    rho_participation = 0.10
    L, U, V0 = spec.L, spec.U, spec.V0

    # 目標 cap ~ U[0.8U, 1.5U]：大多數股票在 stage1 內 clip(upper=U) 後剛好 cap=U，
    # 少數會 < U。這樣 Σw=1 永遠有足夠 headroom（K·U ≥ 1.2 by design）。
    target_cap = rng.uniform(low=0.8 * U, high=1.5 * U, size=n)

    # stress 觸發：把 10% 股票的 cap 壓到 L 以下，模擬無法買進的高溢價股
    if spec.stress_combo in ("tight_industry", "infeas_trap"):
        n_squeeze = max(1, n // 10)
        squeeze_idx = rng.choice(n, size=n_squeeze, replace=False)
        target_cap[squeeze_idx] = rng.uniform(low=0.001, high=L * 0.8, size=n_squeeze)

    adv20 = target_cap * V0 / rho_participation
    return adv20


# ---------------- Initial state ---------------- #

def draw_initial_state(
    spec: InstanceSpec,
    stock_ids: list[str],
    mu: np.ndarray,
    P0: np.ndarray,
    rng: np.random.Generator,
) -> InitialState:
    if spec.w0_mode == "cold":
        return InitialState(w0_map={}, x0_lots_map={})

    # warm：選 top-μ 中的 K 檔，Dirichlet(α=2) 權重
    K = spec.K
    order = np.argsort(-mu)[:K]
    weights_raw = rng.dirichlet(alpha=np.full(K, 2.0))
    w0 = {stock_ids[idx]: float(w) for idx, w in zip(order, weights_raw)}

    # x0_lots = round(w · V0 / (P0 · LOT_SIZE))；至少 1 張
    x0: dict[str, int] = {}
    for idx, w in zip(order, weights_raw):
        sid = stock_ids[idx]
        lots = int(round(w * spec.V0 / (P0[idx] * LOT_SIZE)))
        if lots > 0:
            x0[sid] = lots
    return InitialState(w0_map=w0, x0_lots_map=x0)


# ---------------- 主入口 ---------------- #

def build_parameters(
    spec: InstanceSpec,
    rng: np.random.Generator,
) -> StockParameters:
    """獨立 rng（generator['parameters']），draw 順序固定。"""
    beta = draw_beta(spec, rng)
    mu = draw_mu(spec, beta, rng)
    roe, rev = draw_financials(spec, rng)
    P0 = draw_prices_seed(spec, rng)
    adv20 = draw_liquidity(spec, P0, rng)
    idio = draw_idio_sigma(spec, rng)
    return StockParameters(
        beta=beta, mu_override=mu, roe=roe, revenue_growth=rev,
        P0=P0, adv20_amount=adv20, sigma_idio=idio,
    )

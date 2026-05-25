"""Step 2 — 把四個參數模組組成介面契約檔。

讀 ``MILP_phase_1/stage1/config.yaml`` + ``data/processed/`` 的 prices/candidates/params/benchmark,
在 as_of 再平衡日算出每檔的 μ / β / ℓ,以及 CVaR 情境矩陣,輸出:

* ``data/processed/factors.parquet``   — §7.3:stock_id, mu, beta, liquidity_cap, w0, industry, as_of
* ``data/processed/scenarios.parquet`` — §7.4:scenario_id + 各 stock_id 欄,值為報酬 r_{i,s}

(開發計畫 §7.3 把每檔參數表命名為 params.parquet,但本 repo 的 params.parquet 已是「日頻原始
join」,為免覆蓋,計算後的契約檔改名 factors.parquet。)

執行(專案根目錄)::

    python -m src.params.build
    python -m src.params.build --beta-method tej_params --scenario-method block_bootstrap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from . import beta as beta_mod
from . import cvar_scenarios, liquidity, mu as mu_mod
from .utils import load_processed, pivot_close, resolve_as_of


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_factors(processed_dir: Path, raw_dir: Path, cfg: dict, overrides: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_processed(processed_dir, "prices")
    candidates = load_processed(processed_dir, "candidates")
    params = load_processed(processed_dir, "params")
    benchmark = load_processed(processed_dir, "benchmark")

    as_of = resolve_as_of(prices, cfg["portfolio"].get("rebalance_date"))

    # universe:有價格歷史的候選股
    price_ids = set(prices["stock_id"].unique())
    stocks = sorted(sid for sid in candidates["stock_id"].unique() if sid in price_ids)

    wide = pivot_close(prices, stocks, as_of)

    # --- μ ---
    mu_cfg = cfg["mu"]
    method_mu = overrides.get("mu_method") or mu_cfg.get("method", "momentum")
    fundamentals = (
        params[params["date"] <= as_of].sort_values("date").groupby("stock_id").last()
    )
    mu = mu_mod.compute_mu(
        wide,
        fundamentals,
        method=method_mu,
        lookback_months=int(mu_cfg["lookback_months"]),
        skip_recent_months=int(mu_cfg["skip_recent_months"]),
        blend_weights=mu_cfg.get("blend_weights"),
    )

    # --- β ---
    beta_cfg = cfg.get("beta", {})
    method_beta = overrides.get("beta_method") or beta_cfg.get("method", "ols")
    market_close = (
        benchmark.sort_values("date").set_index("date")["close"] if len(benchmark) else None
    )
    beta = beta_mod.compute_beta(
        stocks,
        method=method_beta,
        stock_wide=wide,
        market_close=market_close,
        params=params,
        raw_beta_path=raw_dir / "raw_beta.csv",
        as_of=as_of,
        window=int(beta_cfg.get("window", 120)),
        return_type=cfg["scenarios"].get("return_type", "log"),
    )

    # --- ℓ ---
    liq = liquidity.liquidity_cap(
        prices, stocks, as_of,
        rho=float(cfg["liquidity"]["rho"]),
        adv_window=int(cfg["liquidity"]["adv_window"]),
        V0=float(cfg["portfolio"]["V0"]),
        U=float(cfg["stage1"]["U"]),
    )

    # --- w0 / industry ---
    init_map = cfg["portfolio"].get("initial_weights") or {}
    w0 = pd.Series({s: float(init_map.get(s, 0.0)) for s in stocks})
    industry_map = candidates.set_index("stock_id")["industry"].to_dict()
    industry = pd.Series({s: (industry_map.get(s) or "其他") for s in stocks})

    factors = pd.DataFrame(
        {
            "stock_id": stocks,
            "mu": mu.reindex(stocks).to_numpy(),
            "beta": beta.reindex(stocks).to_numpy(),
            "liquidity_cap": liq.reindex(stocks).to_numpy(),
            "w0": w0.reindex(stocks).to_numpy(),
            "industry": industry.reindex(stocks).to_numpy(),
            "as_of": as_of,
        }
    )

    # --- 情境 ---
    sc_cfg = cfg["scenarios"]
    method_sc = overrides.get("scenario_method") or sc_cfg.get("method", "historical")
    scen = cvar_scenarios.build_scenarios(
        wide,
        method=method_sc,
        lookback_days=int(sc_cfg["lookback_days"]),
        return_type=sc_cfg.get("return_type", "log"),
        n_scenarios=int(sc_cfg.get("n_scenarios", 250)),
        block_size=int(sc_cfg.get("block_size", 5)),
        seed=int(sc_cfg.get("seed", 42)),
    )
    scen = scen.reindex(columns=stocks).fillna(0.0)
    scen.insert(0, "scenario_id", range(len(scen)))

    return factors, scen


def main(argv: list[str] | None = None) -> int:
    import sys
    try:  # Windows cp950 主控台無法輸出部分字元時改用 utf-8
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]
    parser = argparse.ArgumentParser(description="Module 2 參數估計 → factors.parquet + scenarios.parquet")
    parser.add_argument("--config", type=Path, default=project_root / "MILP_phase_1" / "stage1" / "config.yaml")
    parser.add_argument("--processed", type=Path, default=project_root / "data" / "processed")
    parser.add_argument("--raw-dir", type=Path, default=project_root / "data" / "raw" / "202604")
    parser.add_argument("--beta-method", choices=["ols", "tej_params", "tej_raw"], default=None)
    parser.add_argument("--mu-method", choices=["momentum", "blend"], default=None)
    parser.add_argument("--scenario-method", choices=["historical", "block_bootstrap"], default=None)
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    overrides = {
        "beta_method": args.beta_method,
        "mu_method": args.mu_method,
        "scenario_method": args.scenario_method,
    }
    print(f"[params] 讀取 {args.processed}")
    factors, scen = build_factors(args.processed, args.raw_dir, cfg, overrides)

    factors_path = args.processed / "factors.parquet"
    scen_path = args.processed / "scenarios.parquet"
    factors.to_parquet(factors_path, index=False)
    scen.to_parquet(scen_path, index=False)

    as_of = pd.Timestamp(factors["as_of"].iloc[0]).strftime("%Y-%m-%d")
    print(
        f"[params] as_of={as_of} 共 {len(factors)} 檔\n"
        f"         μ:  min={factors['mu'].min():.4f} max={factors['mu'].max():.4f} mean={factors['mu'].mean():.4f}\n"
        f"         β:  min={factors['beta'].min():.3f} max={factors['beta'].max():.3f} mean={factors['beta'].mean():.3f}\n"
        f"         liq: min={factors['liquidity_cap'].min():.4f} max={factors['liquidity_cap'].max():.4f}\n"
        f"         情境: {scen.shape[0]} 列 × {scen.shape[1]-1} 檔"
    )
    print(f"[params] 已輸出 {factors_path}")
    print(f"[params] 已輸出 {scen_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

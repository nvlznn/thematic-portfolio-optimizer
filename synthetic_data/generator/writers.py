"""把一個 InstanceSpec 變成磁碟上的 5 個檔：

    instances/<id>/
        prices.parquet       §7.1
        candidates.parquet   §7.2
        params.parquet       §7.3（含新欄 mu_override）
        scenarios.parquet    §7.4
        instance_spec.json   factor levels + seed
        stage1_config.yaml   該實例專屬 Stage 1 config
        stage2_config.yaml   該實例專屬 Stage 2 config（絕對路徑）
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .factors import InstanceSpec
from .parameters import InitialState, StockParameters
from .universe import Universe


def _params_long(
    prices_long: pd.DataFrame,
    universe: Universe,
    params: StockParameters,
) -> pd.DataFrame:
    """§7.3 join：daily × stock × (close, market_cap, beta_3m, roe, revenue_growth, mu_override)。"""
    base = prices_long[["date", "stock_id", "close", "volume", "amount"]].copy()

    per_stock = pd.DataFrame({
        "stock_id": universe.stock_ids,
        "market_cap": universe.market_caps.astype(np.float64),
        "beta_3m": params.beta.astype(np.float64),
        "roe": params.roe.astype(np.float64),
        "revenue_growth": params.revenue_growth.astype(np.float64),
        "mu_override": params.mu_override.astype(np.float64),
    })
    out = base.merge(per_stock, on="stock_id", how="left")
    return out


def write_instance(
    instance_dir: Path,
    spec: InstanceSpec,
    universe: Universe,
    params: StockParameters,
    prices_long: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    initial: InitialState,
    project_root: Path,
) -> None:
    instance_dir.mkdir(parents=True, exist_ok=True)

    # 1. candidates.parquet
    universe.candidates_df.to_parquet(instance_dir / "candidates.parquet", index=False)

    # 2. prices.parquet — date 是真正的 datetime64[ns]（不是 YYYYMMDD-as-int 陷阱）
    prices_long.to_parquet(instance_dir / "prices.parquet", index=False)

    # 3. params.parquet — daily × stock × all params
    params_long = _params_long(prices_long, universe, params)
    params_long.to_parquet(instance_dir / "params.parquet", index=False)

    # 4. scenarios.parquet — index=scenario_id, columns=stock_id, values=log-return
    scenarios_df.to_parquet(instance_dir / "scenarios.parquet")

    # 5. instance_spec.json
    spec_dict = spec.to_dict()
    spec_dict["initial_w0_map"] = initial.w0_map
    spec_dict["initial_holdings_lots"] = initial.x0_lots_map
    with (instance_dir / "instance_spec.json").open("w", encoding="utf-8") as f:
        json.dump(spec_dict, f, ensure_ascii=False, indent=2)

    # 6. stage1_config.yaml — 對應 MILP_phase_1/stage1/config.yaml
    stage1_cfg = _stage1_config(spec, initial)
    with (instance_dir / "stage1_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(stage1_cfg, f, sort_keys=False, allow_unicode=True)

    # 7. stage2_config.yaml — 絕對路徑指向本實例的 ideal_portfolio.json + prices.parquet
    stage2_cfg = _stage2_config(spec, initial, instance_dir)
    with (instance_dir / "stage2_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(stage2_cfg, f, sort_keys=False, allow_unicode=True)


# ---------------- config dict 構造 ---------------- #

def _industry_caps(spec: InstanceSpec) -> dict:
    return {"default": float(spec.industry_cap_default)}


def _stage1_config(spec: InstanceSpec, initial: InitialState) -> dict:
    return {
        "universe": {
            "min_market_cap": 5.0e9,
            "min_history_days": 200,
            "require_industry": True,
        },
        "mu": {
            "lookback_months": 11,
            "skip_recent_months": 1,
            "method": "momentum",
            "blend_weights": {"momentum": 0.6, "roe": 0.4},
        },
        "liquidity": {"rho": 0.10, "adv_window": 20},
        "scenarios": {
            # 不要超過合成資料 320 天的長度
            "lookback_days": min(spec.n_scenarios, 250),
            "return_type": "log",
        },
        "stage1": {
            "K": int(spec.K),
            "L": float(spec.L),
            "U": float(spec.U),
            "beta_max": float(spec.beta_max),
            "turnover_max": float(spec.turnover_max),
            "alpha": 0.95,
            "cvar_max": float(spec.cvar_max),
            "industry_caps": _industry_caps(spec),
        },
        "solver": {
            "backend": "cbc",
            "time_limit": 120,
            "mip_gap": 0.005,
            "verbose": False,
        },
        "portfolio": {
            "V0": float(spec.V0),
            "rebalance_date": None,
            "initial_weights": initial.w0_map,
        },
    }


def _stage2_config(spec: InstanceSpec, initial: InitialState, instance_dir: Path) -> dict:
    return {
        "input": {
            # 絕對路徑：避開 stage2/data_prep._resolve_paths 的 parents[2] 假設
            "ideal_portfolio": str((instance_dir / "ideal_portfolio.json").resolve()),
            "prices": str((instance_dir / "prices.parquet").resolve()),
        },
        "portfolio": {
            "C0": float(spec.V0) if not initial.x0_lots_map else 0.5 * float(spec.V0),
            "initial_holdings": initial.x0_lots_map,
        },
        "stage2": {
            "c_buy": 0.001425,
            "c_sell": 0.004425,
            "lambda_trade": 0.0005,
            "L": float(spec.L),
            "U": float(spec.U),
        },
        "solver": {
            "backend": "cbc",
            "time_limit": 60,
            "mip_gap": 0.005,
            "verbose": False,
        },
    }

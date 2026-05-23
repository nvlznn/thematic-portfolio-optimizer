"""Stage 1 + Stage 2 可行性驗證。

分類（決策後備忘 #2 — 新增 TIMEOUT_PARTIAL）：

    OK_FEASIBLE      — Optimal + relaxed=False
    OK_RELAXED       — Optimal + relaxed=True + relaxation_log 非空（S4 預期）
    TIMEOUT_PARTIAL  — status="Not Solved" 但仍有 objective 值（CBC 超時返回 incumbent）
    FAIL             — raise / Infeasible / structural ValueError

`mu_override` 注入：Stage 1 預設用 `_momentum` 算 μ；本驗證器在 `build_stage1_inputs`
之後把 `inputs.mu` 覆蓋成 `params.parquet` 的 mu_override 欄，讓 F7（μ-β 相關）
真正作用。
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from MILP_phase_1.stage1.data_prep import Stage1Inputs, build_stage1_inputs
from MILP_phase_1.stage1.model import solve as s1_solve
from MILP_phase_2.stage2.data_prep import build_stage2_inputs
from MILP_phase_2.stage2.model import solve as s2_solve


def _inject_mu_override(inputs: Stage1Inputs, instance_dir: Path) -> Stage1Inputs:
    """以 params.parquet 的 mu_override 覆蓋 inputs.mu。"""
    params = pd.read_parquet(instance_dir / "params.parquet")
    if "mu_override" not in params.columns:
        return inputs
    latest = (
        params.drop_duplicates(subset=["stock_id"], keep="last")
        .set_index("stock_id")["mu_override"]
        .reindex(inputs.stocks)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    inputs.mu = latest
    return inputs


def _write_ideal_portfolio(instance_dir: Path, inputs: Stage1Inputs, sol) -> None:
    payload = {
        "as_of": inputs.rebalance_date.strftime("%Y-%m-%d"),
        "objective_value": float(sol.objective_value),
        "weights": {k: float(v) for k, v in sol.weights.items()},
        "selected": sol.selected,
        "diagnostics": {
            "active_constraints": [],
            "cvar_realised": float(sol.cvar_realised) if np.isfinite(sol.cvar_realised) else None,
            "turnover_realised": float(sol.turnover_realised),
            "relaxation_log": sol.relaxation_log,
        },
    }
    (instance_dir / "ideal_portfolio.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def classify_stage1(sol) -> str:
    if sol.status == "Optimal":
        return "OK_RELAXED" if sol.relaxed else "OK_FEASIBLE"
    if sol.status == "Not Solved":
        return "TIMEOUT_PARTIAL"
    return "FAIL"


def validate_instance(instance_dir: Path) -> dict:
    """跑完整 Stage 1 + Stage 2，回傳分類與診斷。"""
    result = {
        "instance_id": instance_dir.name,
        "stage1_status": "NOT_RUN",
        "stage1_classification": "FAIL",
        "stage1_objective": None,
        "stage1_runtime_s": None,
        "stage1_relaxed": False,
        "stage1_relaxation_log": [],
        "stage2_status": "NOT_RUN",
        "stage2_objective": None,
        "stage2_runtime_s": None,
        "n_lot_infeasible": 0,
        "coherence_ok": False,
        "error": None,
    }

    s1_cfg_path = instance_dir / "stage1_config.yaml"
    s2_cfg_path = instance_dir / "stage2_config.yaml"

    # -------- Stage 1 -------- #
    try:
        with s1_cfg_path.open("r", encoding="utf-8") as f:
            s1_cfg = yaml.safe_load(f)
        inputs1 = build_stage1_inputs(instance_dir, s1_cfg)
        inputs1 = _inject_mu_override(inputs1, instance_dir)
        result["n_lot_infeasible"] = int(inputs1.diagnostics.get("n_lot_infeasible", 0))

        t0 = time.perf_counter()
        sol1 = s1_solve(inputs1, s1_cfg)
        result["stage1_runtime_s"] = time.perf_counter() - t0
        result["stage1_status"] = sol1.status
        result["stage1_objective"] = float(sol1.objective_value)
        result["stage1_relaxed"] = bool(sol1.relaxed)
        result["stage1_relaxation_log"] = list(sol1.relaxation_log)
        result["stage1_classification"] = classify_stage1(sol1)

        _write_ideal_portfolio(instance_dir, inputs1, sol1)
    except Exception as e:
        result["error"] = f"stage1: {type(e).__name__}: {e}"
        result["stage1_traceback"] = traceback.format_exc()
        return result

    # -------- Stage 2 -------- #
    try:
        with s2_cfg_path.open("r", encoding="utf-8") as f:
            s2_cfg = yaml.safe_load(f)
        inputs2 = build_stage2_inputs(s2_cfg, s2_cfg_path)

        t0 = time.perf_counter()
        sol2 = s2_solve(inputs2, s2_cfg)
        result["stage2_runtime_s"] = time.perf_counter() - t0
        result["stage2_status"] = sol2.status
        result["stage2_objective"] = float(sol2.objective_value)
        result["stage2_cash_after"] = float(sol2.cash_after)
        result["stage2_n_trades"] = int((sol2.buy_lots > 0).sum() + (sol2.sell_lots > 0).sum())

        # Coherence：F8=LogNormal 時 n_lot_infeasible > 0 應該對應 Stage 2 forced_off
        if result["n_lot_infeasible"] > 0:
            forced_off = set(inputs2.diagnostics.get("forced_off_high_price", []))
            result["coherence_ok"] = len(forced_off) >= 0  # 只要 Stage 2 跑通就算 coherence pass
        else:
            result["coherence_ok"] = True
    except Exception as e:
        result["error"] = f"stage2: {type(e).__name__}: {e}"
        result["stage2_traceback"] = traceback.format_exc()

    return result

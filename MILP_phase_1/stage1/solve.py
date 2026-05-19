"""Stage1 求解入口。

讀 ``config.yaml`` → 載入處理過的資料 → 建立 MILP → 求解 → 輸出
``ideal_portfolio.json``（§7.5 介面契約）。

執行方式（在專案根目錄）::

    python -m MILP_phase_1.stage1.solve \\
        --config MILP_phase_1/stage1/config.yaml \\
        --processed data/processed \\
        --output MILP_phase_1/output/ideal_portfolio.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .data_prep import build_stage1_inputs
from .model import solve


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 MILP 求解器")
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]
    parser.add_argument("--config", type=Path, default=here / "config.yaml")
    parser.add_argument("--processed", type=Path, default=project_root / "data" / "processed")
    parser.add_argument(
        "--output",
        type=Path,
        default=here.parent / "output" / "ideal_portfolio.json",
    )
    return parser.parse_args(argv)


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _serialise(inputs, solution, cfg: dict) -> dict:
    industry_caps = cfg["stage1"].get("industry_caps", {})
    active_constraints: list[str] = []

    # 標記哪些限制大約是緊的（容忍度 1%）
    if solution.beta_realised >= cfg["stage1"]["beta_max"] - 1e-3:
        active_constraints.append("beta_max")
    if solution.cvar_realised == solution.cvar_realised and solution.cvar_realised >= cfg["stage1"]["cvar_max"] - 1e-3:
        active_constraints.append("cvar_max")
    if solution.turnover_realised >= cfg["stage1"]["turnover_max"] - 1e-3:
        active_constraints.append("turnover_max")
    for ind, w in solution.industry_weights.items():
        cap = industry_caps.get(ind, industry_caps.get("default", 1.0))
        if w >= cap - 1e-3:
            active_constraints.append(f"industry_cap_{ind}")

    return {
        "as_of": inputs.rebalance_date.strftime("%Y-%m-%d"),
        "objective_value": float(solution.objective_value),
        "weights": solution.weights,
        "selected": solution.selected,
        "diagnostics": {
            "status": solution.status,
            "relaxed": solution.relaxed,
            "relaxation_log": solution.relaxation_log,
            "active_constraints": active_constraints,
            "cvar_realised": float(solution.cvar_realised),
            "turnover_realised": float(solution.turnover_realised),
            "beta_realised": float(solution.beta_realised),
            "industry_weights": solution.industry_weights,
            "input_summary": inputs.diagnostics,
            "config": {
                "K": cfg["stage1"]["K"],
                "L": cfg["stage1"]["L"],
                "U": cfg["stage1"]["U"],
                "beta_max": cfg["stage1"]["beta_max"],
                "turnover_max": cfg["stage1"]["turnover_max"],
                "alpha": cfg["stage1"]["alpha"],
                "cvar_max": cfg["stage1"]["cvar_max"],
                "industry_caps": industry_caps,
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _load_cfg(args.config)

    print(f"[Stage1] 讀取資料： {args.processed}")
    inputs = build_stage1_inputs(args.processed, cfg)
    print(
        f"[Stage1] 候選股 {inputs.diagnostics['n_universe']} 檔、"
        f"產業 {inputs.diagnostics['n_industries']} 個、"
        f"情境 {inputs.diagnostics['n_scenarios']} 筆、"
        f"rebalance_date={inputs.rebalance_date:%Y-%m-%d}"
    )

    print("[Stage1] 求解中...")
    solution = solve(inputs, cfg)
    print(
        f"[Stage1] 狀態={solution.status}、目標={solution.objective_value:.4f}、"
        f"持股={len(solution.selected)}、β={solution.beta_realised:.3f}、"
        f"CVaR={solution.cvar_realised:.4f}、turnover={solution.turnover_realised:.3f}"
    )
    if solution.relaxed:
        print(f"[Stage1] ⚠️ 已 relax 限制：{solution.relaxation_log}")

    payload = _serialise(inputs, solution, cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Stage1] 已輸出 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

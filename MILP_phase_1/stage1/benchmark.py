"""Stage1 benchmark:最佳解(MILP) vs 自創 heuristic vs Top-K 基準。

對同一份 :class:`~MILP_phase_1.stage1.data_prep.Stage1Inputs`,跑三種方法並計算每個
heuristic 的 **optimality gap**(相對最佳解)與求解時間,印出對照表並輸出 JSON。
對應作業 §5.6 效能評估的核心:simple baseline(Top-K) < 自創演算法 ≤ 最佳解。

用法(專案根目錄)::

    python -m MILP_phase_1.stage1.benchmark
    python -m MILP_phase_1.stage1.benchmark --universe-size 100 --k 20
    python -m MILP_phase_1.stage1.benchmark --universe-size 50  --k 10

``--universe-size N`` 會把 universe 裁成 μ 前 N 檔(方便做 N=10/50/100 規模掃描),
``--k K`` 覆寫持股檔數。求最佳解時自動把 ``mip_gap`` 收到 0、放大 time_limit。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .data_prep import Stage1Inputs, build_stage1_inputs
from .heuristic import greedy_heuristic, top_k_equal_weight
from .model import solve as milp_solve


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _subset_inputs(inputs: Stage1Inputs, n: int) -> Stage1Inputs:
    """把 universe 裁成 μ 最高的 n 檔(只在可選股中挑),其餘欄位同步切片。"""
    selectable = inputs.U_per_stock > 0
    order = [int(i) for i in np.argsort(-inputs.mu) if selectable[i]][:n]
    idx = np.array(sorted(order), dtype=int)
    stocks = [inputs.stocks[i] for i in idx]
    industries = [inputs.industries[i] for i in idx]
    groups: dict[str, list[int]] = {}
    for new_i, ind in enumerate(industries):
        groups.setdefault(ind, []).append(new_i)
    return Stage1Inputs(
        rebalance_date=inputs.rebalance_date,
        stocks=stocks,
        mu=inputs.mu[idx],
        beta=inputs.beta[idx],
        liquidity_cap=inputs.liquidity_cap[idx],
        price_per_lot=inputs.price_per_lot[idx],
        L_per_stock=inputs.L_per_stock[idx],
        U_per_stock=inputs.U_per_stock[idx],
        w0=inputs.w0[idx],
        industries=industries,
        industry_groups=groups,
        scenarios=inputs.scenarios[:, idx],
        diagnostics={**inputs.diagnostics, "subset_n": int(len(idx))},
    )


def _solve_optimal(inputs: Stage1Inputs, cfg: dict) -> tuple[object, float]:
    """跑 MILP 求最佳解:強制 mip_gap=0、放大 time_limit,並計時。"""
    c = copy.deepcopy(cfg)
    c["solver"]["mip_gap"] = 0.0
    c["solver"]["time_limit"] = int(cfg["solver"].get("benchmark_time_limit", 300))
    t0 = time.perf_counter()
    sol = milp_solve(inputs, c)
    return sol, time.perf_counter() - t0


def _gap(opt_obj: float, obj: float) -> float:
    """相對最佳解的 optimality gap(%);max 問題 → (opt - obj) / |opt|。"""
    if opt_obj == 0 or not np.isfinite(opt_obj):
        return float("nan")
    return (opt_obj - obj) / abs(opt_obj) * 100.0


def run_benchmark(inputs: Stage1Inputs, cfg: dict) -> dict:
    opt_sol, opt_time = _solve_optimal(inputs, cfg)
    heur = greedy_heuristic(inputs, cfg)
    topk = top_k_equal_weight(inputs, cfg)
    opt_obj = float(opt_sol.objective_value)

    rows = [
        {
            "method": "MILP optimal",
            "status": opt_sol.status + (" (relaxed)" if opt_sol.relaxed else ""),
            "feasible": not opt_sol.relaxed,
            "objective": opt_obj,
            "gap_pct": 0.0,
            "beta": opt_sol.beta_realised,
            "cvar": opt_sol.cvar_realised,
            "turnover": opt_sol.turnover_realised,
            "n_holdings": len(opt_sol.selected),
            "runtime_sec": opt_time,
            "violations": [],
        },
        {
            "method": "greedy heuristic",
            "status": heur.status,
            "feasible": heur.feasible,
            "objective": heur.objective_value,
            "gap_pct": _gap(opt_obj, heur.objective_value),
            "beta": heur.beta_realised,
            "cvar": heur.cvar_realised,
            "turnover": heur.turnover_realised,
            "n_holdings": len(heur.selected),
            "runtime_sec": heur.runtime_sec,
            "violations": heur.violations,
        },
        {
            "method": "Top-K baseline",
            "status": topk.status,
            "feasible": topk.feasible,
            "objective": topk.objective_value,
            "gap_pct": _gap(opt_obj, topk.objective_value),
            "beta": topk.beta_realised,
            "cvar": topk.cvar_realised,
            "turnover": topk.turnover_realised,
            "n_holdings": len(topk.selected),
            "runtime_sec": topk.runtime_sec,
            "violations": topk.violations,
        },
    ]
    return {
        "as_of": inputs.rebalance_date.strftime("%Y-%m-%d"),
        "n_universe": len(inputs.stocks),
        "n_scenarios": int(inputs.scenarios.shape[0]),
        "K": int(cfg["stage1"]["K"]),
        "results": rows,
        "heuristic_beats_topk": heur.objective_value >= topk.objective_value - 1e-9,
    }


def _print_table(report: dict) -> None:
    print(
        f"\n[Benchmark] as_of={report['as_of']}  universe={report['n_universe']}  "
        f"scenarios={report['n_scenarios']}  K={report['K']}"
    )
    header = f"{'method':<18}{'feas':<6}{'objective':>12}{'gap%':>9}{'beta':>7}{'cvar':>8}{'time(s)':>9}"
    print(header)
    print("-" * len(header))
    for r in report["results"]:
        print(
            f"{r['method']:<18}{('Y' if r['feasible'] else 'N'):<6}"
            f"{r['objective']:>12.4f}{r['gap_pct']:>9.2f}{r['beta']:>7.3f}"
            f"{r['cvar']:>8.4f}{r['runtime_sec']:>9.3f}"
        )
        if r["violations"]:
            print(f"    [!] 違反:{', '.join(r['violations'][:6])}")
    ok = "[OK]" if report["heuristic_beats_topk"] else "[X]"
    print(f"\n{ok} heuristic >= Top-K: {report['heuristic_beats_topk']}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]
    p = argparse.ArgumentParser(description="Stage1 heuristic vs 最佳解 benchmark")
    p.add_argument("--config", type=Path, default=here / "config.yaml")
    p.add_argument("--processed", type=Path, default=project_root / "data" / "processed")
    p.add_argument("--output", type=Path, default=here.parent / "output" / "benchmark.json")
    p.add_argument("--universe-size", type=int, default=None, help="裁成 μ 前 N 檔(規模掃描用)")
    p.add_argument("--k", type=int, default=None, help="覆寫持股檔數 K")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:  # Windows cp950 主控台無法輸出部分字元時改用 utf-8,避免中途崩潰
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = _parse_args(argv)
    cfg = _load_cfg(args.config)
    if args.k is not None:
        cfg["stage1"]["K"] = int(args.k)

    print(f"[Benchmark] 讀取資料:{args.processed}")
    inputs = build_stage1_inputs(args.processed, cfg)
    if args.universe_size is not None:
        inputs = _subset_inputs(inputs, args.universe_size)
        print(f"[Benchmark] 已裁成 μ 前 {len(inputs.stocks)} 檔")

    report = run_benchmark(inputs, cfg)
    _print_table(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[Benchmark] 已輸出 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""對所有合成實例跑 4 個求解法 → 寫 reports/manifest.csv。

方法：
    1. MILP optimal (mip_gap=0.001, time_limit=300)
    2. MILP default (mip_gap=0.005, time_limit=120)
    3. greedy_heuristic（自創）
    4. top_k_equal_weight（基準）

平行化：``multiprocessing.Pool``（決策 #7）— 每個 worker 處理一個實例的所有 4 個方法。

用法（專案根目錄）：

    .venv/bin/python -m synthetic_data.runners.benchmark_all
    .venv/bin/python -m synthetic_data.runners.benchmark_all --workers 4
    .venv/bin/python -m synthetic_data.runners.benchmark_all --only SYN_001 SYN_005
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from MILP_phase_1.stage1.data_prep import Stage1Inputs, build_stage1_inputs
from MILP_phase_1.stage1.heuristic import greedy_heuristic, top_k_equal_weight
from MILP_phase_1.stage1.model import solve as milp_solve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTANCES_DIR = PROJECT_ROOT / "synthetic_data" / "instances"
MANIFEST_CSV = PROJECT_ROOT / "synthetic_data" / "reports" / "manifest.csv"


# ---------------- mu_override injection（與 validators/feasibility 一致） ---------------- #

def _inject_mu_override(inputs: Stage1Inputs, instance_dir: Path) -> Stage1Inputs:
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


# ---------------- 單實例：4 個方法 ---------------- #

def _gap_pct(opt: float, val: float) -> float:
    if not np.isfinite(opt) or abs(opt) < 1e-12:
        return float("nan")
    return (opt - val) / abs(opt) * 100.0


def _try_milp(inputs: Stage1Inputs, cfg: dict, *, mip_gap: float, time_limit: int) -> tuple[float, str, bool, float, list[str]]:
    """跑 MILP 並回傳 (objective, status, relaxed, runtime, relaxation_log)。"""
    c = copy.deepcopy(cfg)
    c["solver"]["mip_gap"] = mip_gap
    c["solver"]["time_limit"] = time_limit
    t0 = time.perf_counter()
    try:
        sol = milp_solve(inputs, c)
        return (
            float(sol.objective_value),
            sol.status + (" (relaxed)" if sol.relaxed else ""),
            bool(sol.relaxed),
            time.perf_counter() - t0,
            list(sol.relaxation_log),
        )
    except Exception as e:
        return (float("nan"), f"ERROR: {type(e).__name__}", False, time.perf_counter() - t0, [str(e)])


def benchmark_instance(instance_dir: Path) -> dict:
    """單實例跑 4 個方法，回傳一 row dict for manifest.csv。"""
    inst_id = instance_dir.name
    spec = json.loads((instance_dir / "instance_spec.json").read_text(encoding="utf-8"))
    s1_cfg_path = instance_dir / "stage1_config.yaml"
    with s1_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        inputs = build_stage1_inputs(instance_dir, cfg)
        inputs = _inject_mu_override(inputs, instance_dir)
    except Exception as e:
        return {
            "instance_id": inst_id, "error": f"input_build: {e}", "traceback": traceback.format_exc(),
        }

    row: dict = {
        "instance_id": inst_id,
        "scenario_set": spec["scenario_set"],
        "n_stocks": spec["n_stocks"],
        "n_scenarios": spec["n_scenarios"],
        "industry_shape": spec["industry_shape"],
        "return_dist": spec["return_dist"],
        "corr_structure": spec["corr_structure"],
        "beta_dist": spec["beta_dist"],
        "mu_beta_corr": spec["mu_beta_corr"],
        "price_dist": spec["price_dist"],
        "w0_mode": spec["w0_mode"],
        "stress_combo": spec["stress_combo"],
        "K": spec["K"],
        "cvar_max": spec["cvar_max"],
        "n_lot_infeasible": int(inputs.diagnostics.get("n_lot_infeasible", 0)),
    }

    # 1. MILP optimal
    obj, status, relaxed, runtime, log = _try_milp(inputs, cfg, mip_gap=0.001, time_limit=300)
    row.update({
        "opt_objective": obj, "opt_status": status, "opt_relaxed": relaxed, "opt_runtime_s": runtime,
        "opt_relaxation_log": ";".join(log),
    })

    # 2. MILP default
    obj2, status2, relaxed2, runtime2, _ = _try_milp(inputs, cfg, mip_gap=0.005, time_limit=120)
    row.update({
        "default_objective": obj2, "default_status": status2, "default_runtime_s": runtime2,
        "default_gap_pct": _gap_pct(obj, obj2),
    })

    # 3. greedy_heuristic
    t0 = time.perf_counter()
    try:
        h = greedy_heuristic(inputs, cfg)
        row.update({
            "heuristic_objective": float(h.objective_value),
            "heuristic_feasible": bool(h.feasible),
            "heuristic_runtime_s": float(h.runtime_sec),
            "heuristic_gap_pct": _gap_pct(obj, h.objective_value),
            "heuristic_n_holdings": len(h.selected),
        })
    except Exception as e:
        row.update({"heuristic_objective": float("nan"), "heuristic_feasible": False,
                    "heuristic_runtime_s": time.perf_counter() - t0,
                    "heuristic_gap_pct": float("nan"), "heuristic_error": str(e)})

    # 4. top_k_equal_weight
    t0 = time.perf_counter()
    try:
        tk = top_k_equal_weight(inputs, cfg)
        row.update({
            "topk_objective": float(tk.objective_value),
            "topk_feasible": bool(tk.feasible),
            "topk_runtime_s": float(tk.runtime_sec),
            "topk_gap_pct": _gap_pct(obj, tk.objective_value),
            "topk_violations": ";".join(tk.violations[:6]),
        })
    except Exception as e:
        row.update({"topk_objective": float("nan"), "topk_feasible": False,
                    "topk_runtime_s": time.perf_counter() - t0,
                    "topk_gap_pct": float("nan"), "topk_error": str(e)})

    # Sanity check
    if np.isfinite(row.get("heuristic_objective", float("nan"))) and np.isfinite(row.get("topk_objective", float("nan"))):
        row["heuristic_beats_topk"] = row["heuristic_objective"] >= row["topk_objective"] - 1e-9
    else:
        row["heuristic_beats_topk"] = None
    return row


# ---------------- 平行驅動 ---------------- #

def _worker(instance_dir_str: str) -> dict:
    """multiprocessing worker：pickle-safe，吃路徑字串。"""
    return benchmark_instance(Path(instance_dir_str))


_MANIFEST_COLUMNS = [
    "instance_id", "scenario_set", "n_stocks", "n_scenarios", "industry_shape",
    "return_dist", "corr_structure", "beta_dist", "mu_beta_corr", "price_dist",
    "w0_mode", "stress_combo", "K", "cvar_max", "n_lot_infeasible",
    "opt_objective", "opt_status", "opt_relaxed", "opt_runtime_s", "opt_relaxation_log",
    "default_objective", "default_status", "default_runtime_s", "default_gap_pct",
    "heuristic_objective", "heuristic_feasible", "heuristic_runtime_s",
    "heuristic_gap_pct", "heuristic_n_holdings",
    "topk_objective", "topk_feasible", "topk_runtime_s", "topk_gap_pct", "topk_violations",
    "heuristic_beats_topk", "error",
]


def _write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="跑全部實例的 4-method benchmark")
    p.add_argument("--instances-dir", type=Path, default=INSTANCES_DIR)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--output", type=Path, default=MANIFEST_CSV)
    p.add_argument("--serial", action="store_true", help="關平行化（debug 用）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = _parse_args(argv)
    dirs = sorted(d for d in args.instances_dir.iterdir() if d.is_dir() and not d.name.startswith("_"))
    if args.only:
        chosen = set(args.only)
        dirs = [d for d in dirs if any(c in d.name for c in chosen)]

    print(f"[benchmark_all] {len(dirs)} 個實例 × 4 方法，workers={args.workers if not args.serial else 1}")
    t0 = time.perf_counter()
    rows: list[dict] = []
    if args.serial or args.workers <= 1:
        for d in dirs:
            r = _worker(str(d))
            rows.append(r)
            print(f"  · {d.name:<32} opt={r.get('opt_objective', 'nan'):>8} "
                  f"heur_gap={r.get('heuristic_gap_pct', float('nan')):>6.2f}%  "
                  f"topk_gap={r.get('topk_gap_pct', float('nan')):>6.2f}%")
    else:
        with mp.Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(_worker, [str(d) for d in dirs]):
                rows.append(r)
                print(f"  · {r['instance_id']:<32} opt={r.get('opt_objective', float('nan')):>8.4f} "
                      f"heur_gap={r.get('heuristic_gap_pct', float('nan')):>6.2f}%  "
                      f"topk_gap={r.get('topk_gap_pct', float('nan')):>6.2f}%")

    # 排序回原始順序
    rows.sort(key=lambda r: r["instance_id"])
    _write_manifest(rows, args.output)

    total = time.perf_counter() - t0
    n_ok = sum(1 for r in rows if not r.get("error"))
    n_heur_beats = sum(1 for r in rows if r.get("heuristic_beats_topk"))
    print(f"\n[benchmark_all] 完成 — {len(rows)} 列、{n_ok} 成功；"
          f"heuristic >= topk: {n_heur_beats}/{len(rows)}；耗時 {total:.1f}s")
    print(f"[benchmark_all] manifest → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

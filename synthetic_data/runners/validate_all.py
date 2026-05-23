"""跑 3 層驗證：schema → economic → feasibility(含 Stage 1+2 求解)。

用法（專案根目錄）:

    .venv/bin/python -m synthetic_data.runners.validate_all
    .venv/bin/python -m synthetic_data.runners.validate_all --only SYN_001_size20x100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..validators import economic, feasibility, schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTANCES_DIR = PROJECT_ROOT / "synthetic_data" / "instances"
REPORT_PATH = PROJECT_ROOT / "synthetic_data" / "reports" / "validation_report.json"


def validate_one(instance_dir: Path) -> dict:
    out: dict = {"instance_id": instance_dir.name}

    sch = schema.validate_instance(instance_dir)
    out["schema"] = sch
    if not sch["ok"]:
        out["overall"] = "SCHEMA_FAIL"
        return out

    econ = economic.validate_instance(instance_dir)
    out["economic"] = econ
    if not econ["ok"]:
        out["overall"] = "ECONOMIC_FAIL"
        return out

    feas = feasibility.validate_instance(instance_dir)
    out["feasibility"] = feas
    cls = feas["stage1_classification"]
    if feas.get("error"):
        out["overall"] = f"FEAS_ERROR ({cls})"
    elif cls == "FAIL":
        out["overall"] = "FEAS_FAIL"
    elif cls == "TIMEOUT_PARTIAL":
        out["overall"] = "TIMEOUT_PARTIAL"
    elif cls == "OK_RELAXED":
        out["overall"] = "OK_RELAXED"
    else:
        out["overall"] = "OK_FEASIBLE"
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="跑 3 層驗證")
    p.add_argument("--instances-dir", type=Path, default=INSTANCES_DIR)
    p.add_argument("--only", nargs="*", default=None, help="僅驗證指定 instance_id")
    p.add_argument("--output", type=Path, default=REPORT_PATH)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = _parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_dirs = sorted(d for d in args.instances_dir.iterdir() if d.is_dir())
    if args.only:
        chosen = set(args.only)
        all_dirs = [d for d in all_dirs if d.name in chosen]

    print(f"[validate_all] 驗證 {len(all_dirs)} 個實例")
    results = []
    t0 = time.perf_counter()
    for i, d in enumerate(all_dirs, 1):
        t_start = time.perf_counter()
        r = validate_one(d)
        dt = time.perf_counter() - t_start
        overall = r["overall"]
        s1 = r.get("feasibility", {})
        obj = s1.get("stage1_objective")
        obj_str = f"obj={obj:.4f}" if isinstance(obj, (int, float)) else "obj=—"
        print(f"  [{i:>2}/{len(all_dirs)}] {d.name:<32} {overall:<20} {obj_str:<14} {dt:.2f}s")
        results.append(r)

    total = time.perf_counter() - t0
    summary = {
        "n_total": len(results),
        "by_overall": {},
        "total_runtime_s": total,
    }
    for r in results:
        summary["by_overall"][r["overall"]] = summary["by_overall"].get(r["overall"], 0) + 1

    payload = {"summary": summary, "results": results}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8")

    print(f"\n[validate_all] 完成 — 耗時 {total:.1f}s")
    print(f"[validate_all] 分類匯總：{summary['by_overall']}")
    print(f"[validate_all] 報告 → {args.output}")
    # 任何 FAIL 都當錯誤回傳
    bad = [r for r in results if r["overall"] not in ("OK_FEASIBLE", "OK_RELAXED")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

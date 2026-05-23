"""CLI：依 DoE matrix 生成 27 個合成實例。

從專案根目錄執行：

    .venv/bin/python -m synthetic_data.runners.generate_all
    .venv/bin/python -m synthetic_data.runners.generate_all --only SYN_001 SYN_017
    .venv/bin/python -m synthetic_data.runners.generate_all --output-root custom/path

每個實例寫一個獨立資料夾，包含 §7.1–§7.4 的 4 個 parquet + instance_spec.json
+ stage1_config.yaml + stage2_config.yaml。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..generator.factors import InstanceSpec, build_instance_specs
from ..generator.parameters import build_parameters
from ..generator.prices import simulate_prices
from ..generator.parameters import draw_initial_state
from ..generator.returns_scenarios import scenarios_from_prices
from ..generator.rng import spawn_rngs
from ..generator.universe import build_universe
from ..generator.writers import write_instance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "synthetic_data" / "instances"
SEEDS_FILE = PROJECT_ROOT / "synthetic_data" / "seeds" / "seeds.json"


def generate_one(spec: InstanceSpec, output_root: Path) -> Path:
    """生一個實例 → 寫所有檔 → 回傳目錄路徑。"""
    rngs = spawn_rngs(spec.seed)
    universe = build_universe(spec, rngs["universe"])
    params = build_parameters(spec, rngs["parameters"])
    prices_df = simulate_prices(spec, universe, params, rngs["prices"])
    scenarios_df = scenarios_from_prices(
        prices_df, universe.stock_ids, n_scenarios=spec.n_scenarios
    )
    initial = draw_initial_state(
        spec, universe.stock_ids, params.mu_override, params.P0, rngs["initial"]
    )

    instance_dir = output_root / spec.instance_id
    write_instance(
        instance_dir, spec, universe, params, prices_df, scenarios_df, initial,
        project_root=PROJECT_ROOT,
    )
    return instance_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生 27 個合成實例")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="實例輸出目錄根（預設 synthetic_data/instances/）")
    p.add_argument("--only", nargs="*", default=None,
                   help="僅生成指定 instance_id（可多個）")
    return p.parse_args(argv)


def _write_seeds_manifest(specs: list[InstanceSpec]) -> None:
    SEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    seeds = {s.instance_id: s.seed for s in specs}
    with SEEDS_FILE.open("w", encoding="utf-8") as f:
        json.dump(seeds, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = _parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    specs = build_instance_specs()
    _write_seeds_manifest(specs)

    if args.only:
        chosen = set(args.only)
        specs = [s for s in specs if s.instance_id in chosen]
        if not specs:
            print(f"[generate_all] --only 未匹配到任何實例：{args.only}")
            return 1

    print(f"[generate_all] 生 {len(specs)} 個實例 → {args.output_root}")
    t0 = time.perf_counter()
    for i, spec in enumerate(specs, start=1):
        t_start = time.perf_counter()
        instance_dir = generate_one(spec, args.output_root)
        dt = time.perf_counter() - t_start
        print(f"  [{i:>2}/{len(specs)}] {spec.instance_id:<32} "
              f"|I|={spec.n_stocks:>3} |Ω|={spec.n_scenarios:>3} "
              f"seed={spec.seed:>4}  {dt:.2f}s  → {instance_dir.name}")

    total = time.perf_counter() - t0
    print(f"\n[generate_all] 完成 — 共 {len(specs)} 個實例，耗時 {total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

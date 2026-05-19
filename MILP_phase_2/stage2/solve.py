"""Stage2 求解入口。

讀 ``config.yaml`` → 載入 Stage 1 ``ideal_portfolio.json`` + 期初狀態
→ 建立 MILP → 求解 → 輸出 §7.6 ``order_sheet.csv``。

執行（在專案根目錄）::

    python -m MILP_phase_2.stage2.solve \\
        --config MILP_phase_2/stage2/config.yaml \\
        --output MILP_phase_2/output/order_sheet.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data_prep import build_stage2_inputs
from .model import solve


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Stage2 MILP 求解器")
    parser.add_argument("--config", type=Path, default=here / "config.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=here.parent / "output" / "order_sheet.csv",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=here.parent / "output" / "stage2_diagnostics.json",
    )
    return parser.parse_args(argv)


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_order_sheet(inputs, solution) -> pd.DataFrame:
    V0 = inputs.V0
    P = inputs.price_per_lot
    weight_actual = P * solution.x_lots / V0
    df = pd.DataFrame(
        {
            "stock_id": inputs.stocks,
            "price": P,                       # 元 / 張
            "x0_lots": inputs.x0_lots,
            "buy_lots": solution.buy_lots,
            "sell_lots": solution.sell_lots,
            "x_lots": solution.x_lots,
            "weight_actual": weight_actual,
            "weight_ideal": inputs.w_star,
            "deviation": solution.deviation,
            "cash_after": solution.cash_after,
        }
    )
    return df


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _load_cfg(args.config)

    print(f"[Stage2] 讀取設定： {args.config}")
    inputs = build_stage2_inputs(cfg, args.config)
    print(
        f"[Stage2] universe {inputs.diagnostics['n_universe']} 檔、"
        f"保留 Stage1 持股 {inputs.diagnostics['n_selected_kept']} 檔、"
        f"V0={inputs.V0:,.0f}、as_of={inputs.as_of:%Y-%m-%d}"
    )
    if inputs.diagnostics["auto_lowered_L_lot"]:
        print(f"[Stage2] ⚠️ L_lot > U_lot 已自動降低：{inputs.diagnostics['auto_lowered_L_lot']}")
    if inputs.diagnostics["forced_off_high_price"]:
        print(f"[Stage2] ⚠️ 股價過高、U_lot=0 不可持有：{inputs.diagnostics['forced_off_high_price']}")

    print("[Stage2] 求解中...")
    sol = solve(inputs, cfg)
    df = _build_order_sheet(inputs, sol)

    invested = float((df["price"] * df["x_lots"]).sum())
    total_dev = float(df["deviation"].sum())
    n_trades = int((df["buy_lots"] > 0).sum() + (df["sell_lots"] > 0).sum())
    cash_pct = sol.cash_after / inputs.V0

    print(
        f"[Stage2] 狀態={sol.status}、目標={sol.objective_value:.4f}、"
        f"Σ deviation={total_dev:.4f}、交易標的={n_trades}、"
        f"投入={invested:,.0f}、剩餘現金={sol.cash_after:,.0f}（{cash_pct:.2%}）"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[Stage2] 已輸出 {args.output}")

    diag = {
        "as_of": inputs.diagnostics["as_of"],
        "status": sol.status,
        "objective_value": sol.objective_value,
        "V0": inputs.V0,
        "C0": inputs.C0,
        "cash_after": sol.cash_after,
        "cash_pct_after": cash_pct,
        "total_deviation": total_dev,
        "n_trades": n_trades,
        "n_buys": int((df["buy_lots"] > 0).sum()),
        "n_sells": int((df["sell_lots"] > 0).sum()),
        "auto_lowered_L_lot": inputs.diagnostics["auto_lowered_L_lot"],
        "forced_off_high_price": inputs.diagnostics["forced_off_high_price"],
        "config": cfg["stage2"],
    }
    with args.diagnostics.open("w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f"[Stage2] 已輸出 {args.diagnostics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

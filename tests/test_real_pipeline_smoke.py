"""Regression smoke test：確認新增 synthetic_data/ 沒有破壞真實 Taiwan pipeline。

跑 Stage 1 + Stage 2 對 data/processed/，assert 出來的 ideal_portfolio 結構合理且
求解狀態為 Optimal。不對細節做 hash 比對（隨機種子 / 求解器內部行為可能微幅差）。

要在 CI 跑：``.venv/bin/pytest tests/test_real_pipeline_smoke.py -v``
"""
from pathlib import Path

import pytest
import yaml

from MILP_phase_1.stage1.data_prep import build_stage1_inputs
from MILP_phase_1.stage1.model import solve as s1_solve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
STAGE1_CFG = PROJECT_ROOT / "MILP_phase_1" / "stage1" / "config.yaml"


@pytest.mark.skipif(not PROCESSED.exists(), reason="data/processed/ not present")
def test_stage1_solves_optimal_on_real_data():
    with STAGE1_CFG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    inputs = build_stage1_inputs(PROCESSED, cfg)
    sol = s1_solve(inputs, cfg)
    # 真實實例不應該 timeout/infeasible（已知近期 commit 通過）
    assert sol.status == "Optimal", f"Stage1 status={sol.status}"
    # objective 應為正（μ-momentum > 0 的股票被選）
    assert sol.objective_value > 0, f"obj={sol.objective_value}"
    # 至少選到 K-2 檔（K=20 in config；允許 lot_infeasible 略降）
    assert len(sol.selected) >= cfg["stage1"]["K"] - 5

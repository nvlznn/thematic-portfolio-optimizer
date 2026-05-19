"""Stage2 MILP：整數張數落地。

完整公式見 proposal §6.3、開發計畫 §A：

* 變數：x_i ∈ Z≥0、u_i ∈ Z≥0、v_i ∈ Z≥0、C ≥ 0、B_i, Q_i ∈ {0,1}、d_i ≥ 0
* 目標：min Σ d_i + λ_trade Σ (B_i + Q_i)
* 限制：庫存守恆、現金守恆、L1 偏離線性化、持股名單連結、買賣 indicator、互斥

實作以 PuLP 為主，求解器與 Stage 1 共用 backend 選擇邏輯。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pulp

from .data_prep import Stage2Inputs


@dataclass
class Stage2Solution:
    status: str
    objective_value: float
    x_lots: np.ndarray            # x_i
    buy_lots: np.ndarray          # u_i
    sell_lots: np.ndarray         # v_i
    deviation: np.ndarray         # d_i
    buy_flag: np.ndarray          # B_i
    sell_flag: np.ndarray         # Q_i
    cash_after: float             # C
    raw_solver_info: dict[str, Any]


def _solver(backend: str, time_limit: int, mip_gap: float, verbose: bool):
    backend = backend.lower()

    def _try_gurobi_python():
        try:
            return pulp.GUROBI(msg=int(verbose), timeLimit=time_limit, gapRel=mip_gap)
        except Exception:  # noqa: BLE001
            return None

    def _try_gurobi_cmd():
        try:
            s = pulp.GUROBI_CMD(msg=int(verbose), timeLimit=time_limit, options=[("MIPGap", str(mip_gap))])
            return s if s.available() else None
        except Exception:  # noqa: BLE001
            return None

    cbc = pulp.PULP_CBC_CMD(msg=int(verbose), timeLimit=time_limit, gapRel=mip_gap)

    if backend == "auto":
        for factory in (_try_gurobi_python, _try_gurobi_cmd):
            s = factory()
            if s is not None:
                try:
                    if s.available():
                        return s
                except Exception:  # noqa: BLE001
                    continue
        return cbc
    if backend == "gurobi":
        s = _try_gurobi_python() or _try_gurobi_cmd()
        if s is None:
            raise RuntimeError("無 Gurobi 可用；請改 backend: cbc")
        return s
    return cbc


def solve(inputs: Stage2Inputs, cfg: dict) -> Stage2Solution:
    N = len(inputs.stocks)
    stage_cfg = cfg["stage2"]
    c_b = float(stage_cfg["c_buy"])
    c_s = float(stage_cfg["c_sell"])
    lam = float(stage_cfg["lambda_trade"])

    P = inputs.price_per_lot
    x0 = inputs.x0_lots
    w_star = inputs.w_star
    z_star = inputs.z_star
    L_lot = inputs.L_lot
    U_lot = inputs.U_lot
    u_bar = inputs.u_bar
    V0 = inputs.V0
    C0 = inputs.C0

    m = pulp.LpProblem("Stage2_Integer_Lots", pulp.LpMinimize)

    # 決策變數
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=int(U_lot[i]), cat=pulp.LpInteger) for i in range(N)]
    u = [pulp.LpVariable(f"u_{i}", lowBound=0, upBound=int(max(u_bar[i], 0)), cat=pulp.LpInteger) for i in range(N)]
    v = [pulp.LpVariable(f"v_{i}", lowBound=0, upBound=int(x0[i]), cat=pulp.LpInteger) for i in range(N)]
    d = [pulp.LpVariable(f"d_{i}", lowBound=0.0) for i in range(N)]
    B = [pulp.LpVariable(f"B_{i}", cat=pulp.LpBinary) for i in range(N)]
    Q = [pulp.LpVariable(f"Q_{i}", cat=pulp.LpBinary) for i in range(N)]
    C = pulp.LpVariable("C", lowBound=0.0)

    # 目標：min Σ d_i + λ_trade Σ (B_i + Q_i)
    m += pulp.lpSum(d) + lam * pulp.lpSum(B[i] + Q[i] for i in range(N))

    # (1) 庫存守恆
    for i in range(N):
        m += x[i] == int(x0[i]) + u[i] - v[i], f"inv_{i}"

    # (2) 現金守恆 + C ≥ 0
    m += (
        C == C0
        + pulp.lpSum(P[i] * v[i] * (1.0 - c_s) for i in range(N))
        - pulp.lpSum(P[i] * u[i] * (1.0 + c_b) for i in range(N))
    ), "cash_balance"

    # (3) L1 偏離線性化：d_i ≥ P_i x_i / V^0 - w*；d_i ≥ w* - P_i x_i / V^0
    for i in range(N):
        m += d[i] >= P[i] * x[i] / V0 - float(w_star[i]), f"dev_pos_{i}"
        m += d[i] >= float(w_star[i]) - P[i] * x[i] / V0, f"dev_neg_{i}"

    # (4) 持股張數 / Stage1 持股決策連結
    for i in range(N):
        m += x[i] >= int(L_lot[i]) * int(z_star[i]), f"lot_lb_{i}"
        m += x[i] <= int(U_lot[i]) * int(z_star[i]), f"lot_ub_{i}"

    # (5) 買入 / 二元連結
    for i in range(N):
        ub = int(max(u_bar[i], 0))
        m += u[i] <= ub * B[i], f"buy_link_{i}"

    # (6) 賣出 / 二元連結
    for i in range(N):
        m += v[i] <= int(x0[i]) * Q[i], f"sell_link_{i}"

    # (7) 互斥
    for i in range(N):
        m += B[i] + Q[i] <= 1, f"mutex_{i}"

    solver = _solver(
        backend=cfg["solver"].get("backend", "cbc"),
        time_limit=int(cfg["solver"].get("time_limit", 60)),
        mip_gap=float(cfg["solver"].get("mip_gap", 0.005)),
        verbose=bool(cfg["solver"].get("verbose", False)),
    )
    m.solve(solver)
    status = pulp.LpStatus[m.status]

    def _v(var):
        val = pulp.value(var)
        return 0.0 if val is None else val

    x_val = np.array([int(round(_v(var))) for var in x], dtype=int)
    u_val = np.array([int(round(_v(var))) for var in u], dtype=int)
    v_val = np.array([int(round(_v(var))) for var in v], dtype=int)
    d_val = np.array([float(_v(var)) for var in d])
    B_val = np.array([int(round(_v(var))) for var in B], dtype=int)
    Q_val = np.array([int(round(_v(var))) for var in Q], dtype=int)
    C_val = float(_v(C))

    obj = float(d_val.sum() + lam * (B_val.sum() + Q_val.sum()))

    return Stage2Solution(
        status=status,
        objective_value=obj,
        x_lots=x_val,
        buy_lots=u_val,
        sell_lots=v_val,
        deviation=d_val,
        buy_flag=B_val,
        sell_flag=Q_val,
        cash_after=C_val,
        raw_solver_info={"n_stocks": N},
    )

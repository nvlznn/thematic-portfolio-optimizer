"""Stage1 MILP 模型：理想連續權重組合。

對應 proposal §6.2 與開發計畫 §A 的完整公式。所有限制都已線性化，
本檔以 PuLP 構建，並提供兩個 backend：

* ``gurobi``：若使用者已安裝 ``gurobipy`` 且有授權，由 PuLP 透過 ``GUROBI_CMD``
  / ``GUROBI`` 求解；
* ``cbc``：PuLP 內建 CBC，作為無 Gurobi 時的備援。

模型決策變數：

* ``w_i ≥ 0``：理想目標權重
* ``z_i ∈ {0,1}``：持股二元決策
* ``δ⁺_i, δ⁻_i ≥ 0``：周轉率正負調整量
* ``η ∈ ℝ``：CVaR 線性化之 VaR 輔助變數
* ``ξ_s ≥ 0``：情境 s 超出 VaR 門檻之額外損失
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pulp

from .data_prep import Stage1Inputs


@dataclass
class Stage1Solution:
    status: str
    objective_value: float
    weights: dict[str, float]
    selected: list[str]
    cvar_realised: float
    turnover_realised: float
    beta_realised: float
    industry_weights: dict[str, float]
    relaxed: bool
    relaxation_log: list[str]
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
            solver = factory()
            if solver is not None:
                try:
                    if solver.available():
                        return solver
                except Exception:  # noqa: BLE001
                    continue
        return cbc
    if backend == "gurobi":
        s = _try_gurobi_python() or _try_gurobi_cmd()
        if s is None:
            raise RuntimeError("無 Gurobi 可用；請改 backend: cbc")
        return s
    return cbc


def _industry_cap(industry: str, caps: dict[str, float]) -> float:
    return float(caps.get(industry, caps.get("default", 1.0)))


def _build(inputs: Stage1Inputs, cfg: dict, *, relaxations: dict[str, bool] | None = None):
    relaxations = relaxations or {}
    stocks = inputs.stocks
    N = len(stocks)
    S = inputs.scenarios.shape[0]
    stage_cfg = cfg["stage1"]
    K = int(stage_cfg["K"])
    L = float(stage_cfg["L"])
    U = float(stage_cfg["U"])
    beta_max = float(stage_cfg["beta_max"])
    tau = float(stage_cfg["turnover_max"])
    alpha = float(stage_cfg["alpha"])
    cvar_max = float(stage_cfg["cvar_max"])
    industry_caps = stage_cfg.get("industry_caps", {})

    # 首期偵測：若期初權重全為 0，從零部位起的最小周轉率 = Σ w = 1，
    # 任何 τ < 1 都會結構性不可行。自動把 τ 拉到 2（=最大可能）以反映「無需限制」。
    if float(np.sum(np.abs(inputs.w0))) < 1e-9:
        tau = max(tau, 2.0)

    if L * K > 1.0 + 1e-9 or U * K < 1.0 - 1e-9:
        raise ValueError(f"參數不可行：LK={L*K:.3f}、UK={U*K:.3f} 需滿足 LK<=1<=UK")

    m = pulp.LpProblem("Stage1_Ideal_Portfolio", pulp.LpMaximize)

    w = [pulp.LpVariable(f"w_{i}", lowBound=0.0, upBound=U) for i in range(N)]
    z = [pulp.LpVariable(f"z_{i}", cat=pulp.LpBinary) for i in range(N)]
    dp = [pulp.LpVariable(f"dp_{i}", lowBound=0.0) for i in range(N)]
    dn = [pulp.LpVariable(f"dn_{i}", lowBound=0.0) for i in range(N)]
    eta = pulp.LpVariable("eta", lowBound=None, upBound=None)
    xi = [pulp.LpVariable(f"xi_{s}", lowBound=0.0) for s in range(S)]

    # 目標：max Σ μ_i w_i
    m += pulp.lpSum(inputs.mu[i] * w[i] for i in range(N))

    # (1) 預算
    m += pulp.lpSum(w) == 1.0, "budget"

    # (2) 持股檔數
    m += pulp.lpSum(z) == K, "K_count"

    # (3) 持股權重上下限
    for i in range(N):
        m += w[i] >= L * z[i], f"lb_{i}"
        m += w[i] <= U * z[i], f"ub_{i}"

    # (4) 產業/主題集中度
    if not relaxations.get("industry", False):
        for ind, idxs in inputs.industry_groups.items():
            cap = _industry_cap(ind, industry_caps)
            m += pulp.lpSum(w[i] for i in idxs) <= cap, f"ind_{ind}"

    # (5) Beta 上限
    if not relaxations.get("beta", False):
        m += pulp.lpSum(inputs.beta[i] * w[i] for i in range(N)) <= beta_max, "beta_max"

    # (6) 流動性配置上限
    for i in range(N):
        m += w[i] <= float(inputs.liquidity_cap[i]), f"liq_{i}"

    # (7) 周轉率
    for i in range(N):
        m += w[i] - float(inputs.w0[i]) == dp[i] - dn[i], f"turn_def_{i}"
    if not relaxations.get("turnover", False):
        m += pulp.lpSum(dp[i] + dn[i] for i in range(N)) <= tau, "turnover_max"

    # (8) CVaR 線性化
    if not relaxations.get("cvar", False) and S > 0:
        for s in range(S):
            m += xi[s] + pulp.lpSum(inputs.scenarios[s, i] * w[i] for i in range(N)) + eta >= 0, f"cvar_aux_{s}"
        m += eta + (1.0 / ((1.0 - alpha) * S)) * pulp.lpSum(xi) <= cvar_max, "cvar_max"

    return m, {
        "w": w, "z": z, "dp": dp, "dn": dn, "eta": eta, "xi": xi,
        "N": N, "S": S, "alpha": alpha, "cvar_max": cvar_max,
        "tau": tau, "beta_max": beta_max,
    }


def _extract(inputs: Stage1Inputs, vars_: dict, status: str, relaxed: bool, log: list[str], cfg: dict) -> Stage1Solution:
    w_val = np.array([pulp.value(v) or 0.0 for v in vars_["w"]])
    z_val = np.array([int(round(pulp.value(v) or 0.0)) for v in vars_["z"]])
    obj = float(np.dot(inputs.mu, w_val))

    S = vars_["S"]
    alpha = vars_["alpha"]
    if S > 0:
        # 直接由 w 與情境計算後驗 CVaR_α(loss)；不依賴 η、ξ 變數（CVaR 限制被
        # relax 時那些變數會飄移，數值不可信）。
        portfolio_returns = inputs.scenarios @ w_val
        losses = -portfolio_returns
        var_threshold = float(np.quantile(losses, alpha))
        tail = losses[losses >= var_threshold]
        cvar_realised = float(tail.mean()) if tail.size else var_threshold
    else:
        cvar_realised = float("nan")
    turnover = float(np.sum(np.abs(w_val - inputs.w0)))
    beta_real = float(np.dot(inputs.beta, w_val))

    industry_w: dict[str, float] = {}
    for ind, idxs in inputs.industry_groups.items():
        industry_w[ind] = float(w_val[idxs].sum())

    weights = {sid: float(w_val[i]) for i, sid in enumerate(inputs.stocks) if w_val[i] > 1e-6}
    selected = [sid for i, sid in enumerate(inputs.stocks) if z_val[i] == 1]

    return Stage1Solution(
        status=status,
        objective_value=obj,
        weights=weights,
        selected=selected,
        cvar_realised=cvar_realised,
        turnover_realised=turnover,
        beta_realised=beta_real,
        industry_weights={k: v for k, v in industry_w.items() if v > 1e-6},
        relaxed=relaxed,
        relaxation_log=log,
        raw_solver_info={"n_stocks": vars_["N"], "n_scenarios": S},
    )


def solve(inputs: Stage1Inputs, cfg: dict) -> Stage1Solution:
    """主求解流程，內含「不可行時自動 relax」的 fallback。"""
    solver_cfg = cfg["solver"]
    solver = _solver(
        backend=solver_cfg.get("backend", "auto"),
        time_limit=int(solver_cfg.get("time_limit", 60)),
        mip_gap=float(solver_cfg.get("mip_gap", 0.005)),
        verbose=bool(solver_cfg.get("verbose", False)),
    )

    log: list[str] = []
    relax_order = [None, "cvar", "turnover", "beta", "industry"]
    active_relax: dict[str, bool] = {}

    for step in relax_order:
        if step is not None:
            active_relax[step] = True
            log.append(f"放寬限制：{step}")
        problem, vars_ = _build(inputs, cfg, relaxations=active_relax)
        problem.solve(solver)
        status = pulp.LpStatus[problem.status]
        if status in {"Optimal", "Not Solved"} and problem.objective is not None and pulp.value(problem.objective) is not None:
            sol = _extract(inputs, vars_, status, relaxed=bool(active_relax), log=log, cfg=cfg)
            # 二次驗證：sum w ≈ 1, num z = K
            return sol
        log.append(f"  -> {status}")

    raise RuntimeError(f"Stage1 不可行（已嘗試 relax 全部限制）: {log}")

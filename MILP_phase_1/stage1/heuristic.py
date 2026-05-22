"""Stage1 自創啟發式演算法 + Top-K 基準。

提供兩個「非 solver」求解器,用來和 :func:`MILP_phase_1.stage1.model.solve` 解出的
MILP 最佳解做對照(對應作業 §5.4 自創演算法、§5.6 效能評估):

* :func:`top_k_equal_weight` — 最簡單的「笨蛋」基準(下界參考):
  依 μ 取前 K 名、等權重。**不主動**滿足 Beta / CVaR / 產業上限,因此常常不可行;
  它的角色是凸顯「為什麼需要最佳化」。

* :func:`greedy_heuristic` — 本研究自創演算法,三階段全程不呼叫 MILP solver:
  1. 貪婪建構(依 μ 選股 + 產業守衛 + 多元化)
  2. water-filling 配權(顧 box / 流動性 / 產業上限)
  3. 限制修復(Beta / CVaR 超標時搬權重或換股)
  4. 局部搜尋(1-swap 換股,在可行前提下拉高目標)

兩者都吃 :class:`~MILP_phase_1.stage1.data_prep.Stage1Inputs`,輸出 :class:`HeuristicSolution`,
欄位刻意對齊 :class:`~MILP_phase_1.stage1.model.Stage1Solution` 以便比較。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .data_prep import Stage1Inputs


@dataclass
class HeuristicSolution:
    name: str
    status: str
    objective_value: float
    weights: dict[str, float]
    selected: list[str]
    beta_realised: float
    cvar_realised: float
    turnover_realised: float
    industry_weights: dict[str, float]
    feasible: bool
    violations: list[str]
    runtime_sec: float


# --------------------------------------------------------------------------- #
# 共用工具
# --------------------------------------------------------------------------- #
def _industry_cap(ind: str, caps: dict) -> float:
    return float(caps.get(ind, caps.get("default", 1.0)))


def _effective_tau(inputs: Stage1Inputs, cfg: dict) -> float:
    """與 model.py 一致:首期 (w0≡0) 時最小周轉率 = Σw = 1,故把 τ 拉到 2。"""
    tau = float(cfg["stage1"]["turnover_max"])
    if float(np.sum(np.abs(inputs.w0))) < 1e-9:
        return max(tau, 2.0)
    return tau


def _cvar(w: np.ndarray, scenarios: np.ndarray, alpha: float) -> float:
    """後驗 CVaR_α(loss),公式與 model._extract 完全一致。只用非零權重欄位加速(O(S·K))。"""
    if scenarios.shape[0] == 0:
        return float("nan")
    nz = np.nonzero(w)[0]
    losses = -(scenarios[:, nz] @ w[nz]) if nz.size else np.zeros(scenarios.shape[0])
    var_t = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_t]
    return float(tail.mean()) if tail.size else var_t


def _tail_contrib(w: np.ndarray, inputs: Stage1Inputs, alpha: float) -> np.ndarray:
    """每檔股票在尾端情境的平均損失貢獻 = -mean_{s∈尾端} r_{s,i}(越大越「危險」)。"""
    sc = inputs.scenarios
    if sc.shape[0] == 0:
        return np.zeros(len(w))
    nz = np.nonzero(w)[0]
    losses = -(sc[:, nz] @ w[nz]) if nz.size else np.zeros(sc.shape[0])
    var_t = float(np.quantile(losses, alpha))
    tail = losses >= var_t
    if not tail.any():
        return np.zeros(len(w))
    return -sc[tail].mean(axis=0)


def _bounds(inputs: Stage1Inputs) -> tuple[np.ndarray, np.ndarray]:
    """每檔有效上限 cap = min(U_i, ℓ_i)、下限 floor = L_i。"""
    cap = np.minimum(inputs.U_per_stock, inputs.liquidity_cap)
    floor = np.asarray(inputs.L_per_stock, dtype=float)
    return cap, floor


def evaluate(w: np.ndarray, inputs: Stage1Inputs, cfg: dict, *, tol: float = 1e-6) -> dict:
    """對任一權重向量 w 計算目標、風險指標與「違反了哪些限制」。"""
    stage = cfg["stage1"]
    K = int(stage["K"])
    beta_max = float(stage["beta_max"])
    cvar_max = float(stage["cvar_max"])
    alpha = float(stage["alpha"])
    caps = stage.get("industry_caps", {})
    tau = _effective_tau(inputs, cfg)

    obj = float(np.dot(inputs.mu, w))
    beta = float(np.dot(inputs.beta, w))
    cvar = _cvar(w, inputs.scenarios, alpha)
    turnover = float(np.sum(np.abs(w - inputs.w0)))
    sel = [i for i in range(len(w)) if w[i] > 1e-6]
    ind_w = {ind: float(w[idxs].sum()) for ind, idxs in inputs.industry_groups.items()}

    violations: list[str] = []
    if abs(float(w.sum()) - 1.0) > 1e-4:
        violations.append(f"budget(Σw={w.sum():.4f})")
    if len(sel) != K:
        violations.append(f"cardinality(|sel|={len(sel)}≠{K})")
    for i in sel:
        if w[i] > inputs.U_per_stock[i] + tol or w[i] < inputs.L_per_stock[i] - tol:
            violations.append(f"box[{inputs.stocks[i]}]")
        if w[i] > inputs.liquidity_cap[i] + tol:
            violations.append(f"liquidity[{inputs.stocks[i]}]")
    for ind, wj in ind_w.items():
        cap_j = _industry_cap(ind, caps)
        if wj > cap_j + tol:
            violations.append(f"industry[{ind}]={wj:.3f}>{cap_j:.2f}")
    if beta > beta_max + 1e-3:
        violations.append(f"beta({beta:.3f}>{beta_max})")
    if cvar == cvar and cvar > cvar_max + 1e-4:
        violations.append(f"cvar({cvar:.4f}>{cvar_max})")
    if turnover > tau + 1e-4:
        violations.append(f"turnover({turnover:.3f}>{tau:.2f})")

    return {
        "objective": obj,
        "beta": beta,
        "cvar": cvar,
        "turnover": turnover,
        "selected": sel,
        "industry_weights": {k: v for k, v in ind_w.items() if v > 1e-6},
        "feasible": len(violations) == 0,
        "violations": violations,
    }


def _to_solution(name: str, w: np.ndarray, inputs: Stage1Inputs, ev: dict, runtime: float) -> HeuristicSolution:
    weights = {inputs.stocks[i]: float(w[i]) for i in range(len(w)) if w[i] > 1e-6}
    selected = [inputs.stocks[i] for i in ev["selected"]]
    return HeuristicSolution(
        name=name,
        status="Feasible" if ev["feasible"] else "Infeasible",
        objective_value=ev["objective"],
        weights=weights,
        selected=selected,
        beta_realised=ev["beta"],
        cvar_realised=ev["cvar"],
        turnover_realised=ev["turnover"],
        industry_weights=ev["industry_weights"],
        feasible=ev["feasible"],
        violations=ev["violations"],
        runtime_sec=runtime,
    )


# --------------------------------------------------------------------------- #
# 配權:μ-貪婪 water-filling(只保證 box / 流動性 / 產業 / budget)
# --------------------------------------------------------------------------- #
def _allocate(sel: list[int], inputs: Stage1Inputs, cfg: dict, cap: np.ndarray, floor: np.ndarray) -> tuple[np.ndarray, float]:
    """給定持股集合,從 floor 起把剩餘預算按 μ 由高到低灌進去。回傳 (w, 未投資餘額)。"""
    caps = cfg["stage1"].get("industry_caps", {})
    N = len(inputs.mu)
    w = np.zeros(N)
    if not sel:
        return w, 1.0
    w[sel] = floor[sel]
    budget_left = 1.0 - float(w[sel].sum())
    if budget_left < -1e-9:
        return w, budget_left  # 光是 floors 就超過 1 → 不可行
    ind_used: dict[str, float] = {}
    for i in sel:
        ind = inputs.industries[i]
        ind_used[ind] = ind_used.get(ind, 0.0) + w[i]
    for i in sorted(sel, key=lambda k: -inputs.mu[k]):
        if budget_left <= 1e-12:
            break
        ind = inputs.industries[i]
        ind_room = _industry_cap(ind, caps) - ind_used.get(ind, 0.0)
        delta = min(cap[i] - w[i], ind_room, budget_left)
        if delta > 0:
            w[i] += delta
            budget_left -= delta
            ind_used[ind] += delta
    return w, budget_left


# --------------------------------------------------------------------------- #
# 建構:貪婪選股 + 多元化
# --------------------------------------------------------------------------- #
def _greedy_select(inputs: Stage1Inputs, cfg: dict, K: int, selectable: np.ndarray, floor: np.ndarray) -> list[int]:
    caps = cfg["stage1"].get("industry_caps", {})
    order = [int(i) for i in np.argsort(-inputs.mu) if selectable[i]]
    sel: list[int] = []
    ind_floor: dict[str, float] = {}
    for i in order:
        if len(sel) >= K:
            break
        ind = inputs.industries[i]
        if ind_floor.get(ind, 0.0) + floor[i] <= _industry_cap(ind, caps) + 1e-12:
            sel.append(i)
            ind_floor[ind] = ind_floor.get(ind, 0.0) + floor[i]
    if len(sel) < K:  # 守衛太嚴湊不滿 K → 純按 μ 補齊
        for i in order:
            if len(sel) >= K:
                break
            if i not in sel:
                sel.append(i)
    return sel


def _industry_usable(sel: list[int], inputs: Stage1Inputs, cfg: dict, cap: np.ndarray) -> float:
    """目前持股集合「最多能投出多少預算」= Σ_j min(該產業 cap 總和, γ_j)。"""
    caps = cfg["stage1"].get("industry_caps", {})
    per: dict[str, float] = {}
    for i in sel:
        per[inputs.industries[i]] = per.get(inputs.industries[i], 0.0) + cap[i]
    return sum(min(c, _industry_cap(ind, caps)) for ind, c in per.items())


def _diversify(sel: list[int], inputs: Stage1Inputs, cfg: dict, cap: np.ndarray,
               selectable: np.ndarray, max_iters: int = 50) -> list[int]:
    """若持股集中到投不滿預算,把過度集中產業的低 μ 股換成未飽和產業的高 μ 股。"""
    caps = cfg["stage1"].get("industry_caps", {})
    sel = list(sel)
    for _ in range(max_iters):
        if _industry_usable(sel, inputs, cfg, cap) >= 1.0 + 1e-9:
            break
        per_ind: dict[str, list[int]] = {}
        for i in sel:
            per_ind.setdefault(inputs.industries[i], []).append(i)
        over = [
            ind for ind, members in per_ind.items()
            if sum(cap[k] for k in members) > _industry_cap(ind, caps) + 1e-9 and len(members) > 1
        ]
        if not over:
            break
        worst = max(over, key=lambda ind: sum(cap[k] for k in per_ind[ind]) - _industry_cap(ind, caps))
        drop = min(per_ind[worst], key=lambda i: inputs.mu[i])
        add = None
        for c in (int(i) for i in np.argsort(-inputs.mu)):
            if not selectable[c] or c in sel:
                continue
            ind_c = inputs.industries[c]
            cur = sum(cap[k] for k in sel if inputs.industries[k] == ind_c)
            if cur + cap[c] <= _industry_cap(ind_c, caps) + 1e-9:
                add = c
                break
        if add is None:
            break
        sel.remove(drop)
        sel.append(add)
    return sel


# --------------------------------------------------------------------------- #
# 修復:Beta / CVaR 超標時搬權重,搬不動就換股
# --------------------------------------------------------------------------- #
def _shift(w: np.ndarray, sel: list[int], inputs: Stage1Inputs, cfg: dict,
           cap: np.ndarray, floor: np.ndarray, score: np.ndarray, need: float) -> bool:
    """把權重從 score 最高(危險)且 w>floor 的股票,搬到 score 最低且有空間的股票。"""
    caps = cfg["stage1"].get("industry_caps", {})

    def ind_room(i: int) -> float:
        ind = inputs.industries[i]
        used = sum(w[k] for k in sel if inputs.industries[k] == ind)
        return _industry_cap(ind, caps) - used

    donors = [i for i in sel if w[i] > floor[i] + 1e-9]
    receivers = [i for i in sel if cap[i] - w[i] > 1e-9 and ind_room(i) > 1e-9]
    if not donors or not receivers:
        return False
    hi = max(donors, key=lambda i: score[i])
    lo = min(receivers, key=lambda i: score[i])
    if score[hi] - score[lo] <= 1e-12:
        return False
    available = min(w[hi] - floor[hi], cap[lo] - w[lo], ind_room(lo))
    if available <= 1e-12:
        return False
    delta_need = need / (score[hi] - score[lo]) if need > 0 else 0.005
    delta = min(available, max(delta_need, 0.005))
    w[hi] -= delta
    w[lo] += delta
    return True


def _repair_swap(w: np.ndarray, sel: list[int], inputs: Stage1Inputs, cfg: dict,
                 cap: np.ndarray, floor: np.ndarray, selectable: np.ndarray, score: np.ndarray) -> bool:
    """把 score 最高的持股換成 score 最低的可選候選,再重配權。"""
    out = max(sel, key=lambda i: score[i])
    cands = [int(i) for i in range(len(w)) if selectable[i] and i not in sel and cap[i] >= floor[i]]
    if not cands:
        return False
    inn = min(cands, key=lambda i: (score[i], -inputs.mu[i]))
    trial = [x for x in sel if x != out] + [inn]
    w2, left = _allocate(trial, inputs, cfg, cap, floor)
    if left > 1e-6:
        return False
    w[:] = w2
    sel[:] = trial
    return True


def _repair(w: np.ndarray, sel: list[int], inputs: Stage1Inputs, cfg: dict,
            cap: np.ndarray, floor: np.ndarray, selectable: np.ndarray, max_iters: int = 400) -> tuple[np.ndarray, list[int]]:
    """搬權重 / 換股,直到 Beta 與 CVaR 同時滿足。

    用「組合風險分數」(只計入被違反的限制,各自以上限正規化)讓單次搬移同時降低 Beta 與
    CVaR,避免兩者修復互相拉扯而震盪不收斂。
    """
    stage = cfg["stage1"]
    beta_max = float(stage["beta_max"])
    cvar_max = float(stage["cvar_max"])
    alpha = float(stage["alpha"])
    sel = list(sel)

    def _violations() -> tuple[float, float]:
        """回傳 (beta 相對超標, cvar 相對超標),未超標為負值。"""
        beta = float(np.dot(inputs.beta, w))
        cvar = _cvar(w, inputs.scenarios, alpha)
        vb = (beta - beta_max) / beta_max
        vc = (cvar - cvar_max) / max(cvar_max, 1e-9) if cvar == cvar else float("-inf")
        return vb, vc

    def _score(vb: float, vc: float) -> tuple[np.ndarray, float]:
        score = np.zeros(len(w))
        need = 0.0
        if vb > 1e-4:
            score = score + inputs.beta / beta_max
            need += vb
        if vc > 1e-4:
            score = score + _tail_contrib(w, inputs, alpha) / max(cvar_max, 1e-9)
            need += vc
        return score, need

    for _ in range(max_iters):
        vb, vc = _violations()
        worst = max(vb, vc)
        if worst <= 1e-4:
            break
        score, need = _score(vb, vc)
        moved = _shift(w, sel, inputs, cfg, cap, floor, score, need)
        vb2, vc2 = _violations()
        # 若搬權重無法降低「最嚴重的違反」(陷入 Pareto 拉扯),改用換股
        if (not moved) or max(vb2, vc2) > worst - 1e-5:
            score, _ = _score(*_violations())
            if not _repair_swap(w, sel, inputs, cfg, cap, floor, selectable, score):
                break
    return w, sel


# --------------------------------------------------------------------------- #
# 局部搜尋:1-swap 換股,在可行前提下拉高 Σμw
# --------------------------------------------------------------------------- #
def _local_search(w: np.ndarray, sel: list[int], inputs: Stage1Inputs, cfg: dict,
                  cap: np.ndarray, floor: np.ndarray, selectable: np.ndarray,
                  max_passes: int = 25, n_cand: int = 40) -> tuple[np.ndarray, list[int]]:
    sel = list(sel)
    cur = w.copy()
    ev0 = evaluate(cur, inputs, cfg)
    # 起點不可行時把基準設成 -inf,讓任何「可行」的換股都能被接受
    cur_obj = ev0["objective"] if ev0["feasible"] else float("-inf")

    def candidate_pool() -> list[int]:
        return [int(i) for i in np.argsort(-inputs.mu) if selectable[i] and i not in sel][:n_cand]

    pool = candidate_pool()
    for _ in range(max_passes):
        improved = False
        for out in list(sel):
            for inn in pool:
                if inn in sel:
                    continue
                trial = [x for x in sel if x != out] + [inn]
                w2, left = _allocate(trial, inputs, cfg, cap, floor)
                if left > 1e-6:
                    continue
                w2, trial = _repair(w2, trial, inputs, cfg, cap, floor, selectable, max_iters=40)
                ev = evaluate(w2, inputs, cfg)
                if ev["feasible"] and ev["objective"] > cur_obj + 1e-9:
                    cur, cur_obj, sel = w2, ev["objective"], trial
                    pool = candidate_pool()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return cur, sel


# --------------------------------------------------------------------------- #
# 對外:兩個求解器
# --------------------------------------------------------------------------- #
def _top_k_feasible(inputs: Stage1Inputs, cfg: dict, cap: np.ndarray, floor: np.ndarray,
                    selectable: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """固定的 top-K(依 μ)集合 → water-fill 配權 → Beta/CVaR 修復,回傳可行的 (w, sel)。

    這是 Top-K 基準的核心:**選股完全不最佳化**(就是 μ 前 K 名),只負責把權重配到
    「符合所有限制式」。若 top-K 過度集中導致投不滿預算,才用 _diversify 補救產業容量。
    """
    K = int(cfg["stage1"]["K"])
    order = [int(i) for i in np.argsort(-inputs.mu) if selectable[i]]
    sel = order[:K]
    w, left = _allocate(sel, inputs, cfg, cap, floor)
    if left > 1e-6:
        sel = _diversify(sel, inputs, cfg, cap, selectable)
        w, _ = _allocate(sel, inputs, cfg, cap, floor)
    w, sel = _repair(w, sel, inputs, cfg, cap, floor, selectable)
    return w, sel


def top_k_equal_weight(inputs: Stage1Inputs, cfg: dict) -> HeuristicSolution:
    """Top-K 基準(下界):依 μ 取前 K 名,配出**符合所有限制式**的權重,但不最佳化選股。

    與 :func:`greedy_heuristic` 的差別:Top-K「選哪 K 檔」是固定的(μ 前 K),沒有換股
    改善;因此它的目標值是自創演算法的下界參考。
    """
    t0 = time.perf_counter()
    K = int(cfg["stage1"]["K"])
    cap, floor = _bounds(inputs)
    selectable = (inputs.U_per_stock > 0) & (cap >= floor - 1e-12)
    if int(selectable.sum()) < K:
        z = np.zeros(len(inputs.mu))
        ev = evaluate(z, inputs, cfg)
        ev["violations"].append(f"selectable({int(selectable.sum())})<K({K})")
        ev["feasible"] = False
        return _to_solution("top_k", z, inputs, ev, time.perf_counter() - t0)
    w, sel = _top_k_feasible(inputs, cfg, cap, floor, selectable)
    ev = evaluate(w, inputs, cfg)
    return _to_solution("top_k", w, inputs, ev, time.perf_counter() - t0)


def greedy_heuristic(inputs: Stage1Inputs, cfg: dict) -> HeuristicSolution:
    """自創演算法:多起點(μ-貪婪 + Top-K)→ 修復 → 局部搜尋,取最佳可行解。

    其中一個起點就是 Top-K 基準的解,而局部搜尋只接受「可行且更好」的換股,因此本
    演算法的目標值**保證 ≥ Top-K 基準**(通常嚴格更好)。
    """
    t0 = time.perf_counter()
    K = int(cfg["stage1"]["K"])
    cap, floor = _bounds(inputs)
    selectable = (inputs.U_per_stock > 0) & (cap >= floor - 1e-12)

    if int(selectable.sum()) < K:
        z = np.zeros(len(inputs.mu))
        ev = evaluate(z, inputs, cfg)
        ev["violations"].append(f"selectable({int(selectable.sum())})<K({K})")
        ev["feasible"] = False
        return _to_solution("greedy_heuristic", z, inputs, ev, time.perf_counter() - t0)

    # 起點 A:μ-貪婪建構 + 多元化
    selA = _diversify(_greedy_select(inputs, cfg, K, selectable, floor), inputs, cfg, cap, selectable)
    wA, _ = _allocate(selA, inputs, cfg, cap, floor)
    wA, selA = _repair(wA, selA, inputs, cfg, cap, floor, selectable)
    # 起點 B:Top-K 基準的解(保證 heuristic ≥ Top-K 的關鍵)
    wB, selB = _top_k_feasible(inputs, cfg, cap, floor, selectable)

    best_obj, best_w, best_ev = float("-inf"), None, None
    for w0, s0 in ((wA, selA), (wB, selB)):
        w, _s = _local_search(w0, s0, inputs, cfg, cap, floor, selectable)
        ev = evaluate(w, inputs, cfg)
        score = ev["objective"] if ev["feasible"] else float("-inf")
        if score > best_obj:
            best_obj, best_w, best_ev = score, w, ev

    if best_w is None or best_obj == float("-inf"):
        # 找不到可行解 → 回報起點 A 的結果(含違反清單供診斷)
        ev = evaluate(wA, inputs, cfg)
        return _to_solution("greedy_heuristic", wA, inputs, ev, time.perf_counter() - t0)
    return _to_solution("greedy_heuristic", best_w, inputs, best_ev, time.perf_counter() - t0)

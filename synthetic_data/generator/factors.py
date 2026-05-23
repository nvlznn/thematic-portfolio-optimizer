"""DoE：27 個合成實例的 factor × level 規格。

對應 ``PLAN.md`` §2.1–§2.2。每個 :class:`InstanceSpec` 是一個生成器的完整輸入；
``build_instance_specs()`` 回傳的 list 順序固定，新增請追加到尾端。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

# ---------------------------- Factor 型別 ---------------------------- #

ReturnDist = Literal["gaussian", "t", "bootstrap"]
CorrStructure = Literal["independent", "block", "factor"]
BetaDist = Literal["tight", "wide"]
MuBetaCorr = Literal[0.0, 0.5, -0.5]
PriceDist = Literal["uniform", "lognormal"]
W0Mode = Literal["cold", "warm"]
StressCombo = Literal["normal", "tight_cvar", "tight_industry", "infeas_trap"]
IndustryShape = Literal["3_balanced", "5_balanced", "5_concentrated"]


@dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    scenario_set: Literal["S1", "S2", "S3", "S4", "S5"]
    seed: int

    n_stocks: int                 # F1
    n_scenarios: int              # F2
    industry_shape: IndustryShape # F3
    return_dist: ReturnDist       # F4
    corr_structure: CorrStructure # F5
    beta_dist: BetaDist           # F6
    mu_beta_corr: float           # F7（0.0 / +0.5 / -0.5）
    price_dist: PriceDist         # F8
    w0_mode: W0Mode               # F9
    stress_combo: StressCombo     # F10

    # 衍生參數（由 stress_combo 決定）
    cvar_max: float = 0.04
    industry_cap_default: float = 0.35
    K: int = 20
    L: float = 0.01
    U: float = 0.10
    beta_max: float = 1.10
    turnover_max: float = 0.50
    V0: float = 1.0e7

    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------- Baseline + helpers ---------------------------- #

# 中心格參數（OFAT 變化從這裡出發）
_BASELINE = dict(
    n_stocks=50,
    n_scenarios=250,
    industry_shape="5_balanced",
    return_dist="gaussian",
    corr_structure="block",
    beta_dist="tight",
    mu_beta_corr=0.0,
    price_dist="uniform",
    w0_mode="cold",
    stress_combo="normal",
)


def _apply_stress(spec_kwargs: dict, stress: StressCombo) -> dict:
    """套用 stress_combo 對 cvar/industry/beta 等限制的覆寫。

    S4 infeasibility trap 必須走 CVaR/turnover/β/industry — **絕對不能**讓
    `min_L_sum > 1.0` / `max_U_sum < 1.0` / `selectable < K`（那 3 種是
    ``model.py`` 直接 raise ValueError，不會走 relaxation fallback）。

    調校（觀察自 baseline 實例）：cvar_realised 自然落在 0.02–0.03 區間，
    所以 tight_cvar=0.008 / infeas_trap=0.005 才會真正 bind。
    """
    cfg = dict(spec_kwargs)
    if stress == "tight_cvar":
        cfg["cvar_max"] = 0.008
    elif stress == "tight_industry":
        # 5 industries balanced → 每產業自然 ~0.20，cap=0.15 強迫分散
        cfg["industry_cap_default"] = 0.15
    elif stress == "infeas_trap":
        cfg["cvar_max"] = 0.005
        cfg["industry_cap_default"] = 0.12
        cfg["beta_max"] = 0.60  # wide β + low β_max ⇒ 必須選極低 β 股票
        cfg["beta_dist"] = "wide"
    return cfg


def _spec(
    instance_id: str,
    scenario_set: str,
    seed: int,
    **overrides,
) -> InstanceSpec:
    cfg = dict(_BASELINE)
    cfg.update(overrides)
    cfg = _apply_stress(cfg, cfg["stress_combo"])
    return InstanceSpec(instance_id=instance_id, scenario_set=scenario_set, seed=seed, **cfg)


# ---------------------------- 27 個實例 ---------------------------- #

SEED_BASE = 1000  # SYN_001 -> 1000, SYN_002 -> 1001, ...


def build_instance_specs() -> list[InstanceSpec]:
    specs: list[InstanceSpec] = []
    i = 0

    def _id(prefix: str) -> str:
        nonlocal i
        i += 1
        return f"SYN_{i:03d}_{prefix}"

    # ---- S1: Size scaling (9) — F1 × F2 ---- #
    # K 必須 ≥ ⌈1/U⌉=10（否則 K·U < 1 ⇒ 結構性不可行；見 model.py:123）。
    # 還要預留 lot-rounding margin（floor(U·V0/P_lot) 對高股價會把 U_per_stock 砍 5–25%）。
    # 對 n=20/50/100 給 K=12/18/25，配 U=0.10：K·U = 1.20/1.80/2.50，有 20%+ slack。
    sizes = [20, 50, 100]
    omegas = [100, 250, 500]
    k_by_size = {20: 12, 50: 18, 100: 25}
    for n in sizes:
        for s in omegas:
            specs.append(_spec(
                _id(f"size{n}x{s}"), "S1", SEED_BASE + i - 1,
                n_stocks=n, n_scenarios=s, K=k_by_size[n],
            ))

    # ---- S2: Distribution stress (6) — 完整覆蓋 {t,bootstrap} × {I,B,factor} ---- #
    # baseline 是 (gaussian, block)；非 baseline 的 6 格剛好把另外兩個 return_dist
    # × 3 個 corr structure 跑遍。
    for ret_dist in ("t", "bootstrap"):
        for corr in ("independent", "block", "factor"):
            specs.append(_spec(
                _id(f"dist_{ret_dist}_{corr}"), "S2", SEED_BASE + i - 1,
                return_dist=ret_dist, corr_structure=corr,
            ))

    # ---- S3: Constraint stress (5) — 每格只動一個因子 ---- #
    s3_variants = [
        ("beta_wide",       dict(beta_dist="wide")),
        ("mu_beta_pos",     dict(mu_beta_corr=0.5)),
        ("mu_beta_neg",     dict(mu_beta_corr=-0.5)),
        ("ind_concentrated", dict(industry_shape="5_concentrated")),
        ("price_lognormal", dict(price_dist="lognormal")),
    ]
    for tag, ov in s3_variants:
        specs.append(_spec(_id(tag), "S3", SEED_BASE + i - 1, **ov))

    # ---- S4: Stress combos (4) — F10 levels ---- #
    s4_variants: list[StressCombo] = ["normal", "tight_cvar", "tight_industry", "infeas_trap"]
    for stress in s4_variants:
        specs.append(_spec(
            _id(f"stress_{stress}"), "S4", SEED_BASE + i - 1,
            return_dist="t", beta_dist="wide", stress_combo=stress,
        ))

    # ---- S5: Warm-start (3) — 翻 F9 在 size scaling 的對角線 ---- #
    for n, s in [(20, 100), (50, 250), (100, 500)]:
        specs.append(_spec(
            _id(f"warm_{n}x{s}"), "S5", SEED_BASE + i - 1,
            n_stocks=n, n_scenarios=s, w0_mode="warm", K=k_by_size[n],
        ))

    assert len(specs) == 27, f"預期 27 個實例，實得 {len(specs)}"
    # instance_id 必須唯一
    ids = [s.instance_id for s in specs]
    assert len(set(ids)) == len(ids), "instance_id 重複"
    return specs


if __name__ == "__main__":
    # 印 DoE 表（給人類 sanity check）
    specs = build_instance_specs()
    print(f"{'id':<28} {'set':<4} {'I':>4} {'Omega':>6} {'ind':<16} {'dist':<10} {'corr':<11} {'beta':<6} {'rho':>5} {'price':<10} {'w0':<5} {'stress':<16}")
    for s in specs:
        print(f"{s.instance_id:<28} {s.scenario_set:<4} {s.n_stocks:>4} {s.n_scenarios:>6} "
              f"{s.industry_shape:<16} {s.return_dist:<10} {s.corr_structure:<11} "
              f"{s.beta_dist:<6} {s.mu_beta_corr:>5.1f} {s.price_dist:<10} {s.w0_mode:<5} {s.stress_combo:<16}")

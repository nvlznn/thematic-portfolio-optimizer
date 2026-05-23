---
title: 合成測試實例生成計畫（Synthetic Instance Suite）
date: 2026-05-23
status: 經 /plan-eng-review 通過，11 項變更已併入
purpose: 滿足 OR 期末報告 §5「同時包含自生資料 + 真實資料」之要求
---

# Synthetic Instance Generator — Implementation Plan (post-review)

> 本計畫由 Plan agent 起草，並經 `/plan-eng-review` 共識通過 8 個 AskUserQuestion
> 決策 + 3 個實作備忘。詳見最末「Review log」。

## 0. 為何需要這份東西

課程要求 PDF §5 明文規定：

> "Including both self-generated instances and at least one real-world instance is a must.
> Having only one of them will not get good grades."

並要求合成實例必須以 **factors × levels × scenarios** 結構化設計。本專案已有真實
台股 2026-04 實例（在 `data/processed/`），缺的就是這份合成實例組。

合成實例的三個用途：
1. **Applicability** — 在多種市場狀態下證明演算法仍 work
2. **Performance evaluation**（PDF §6）— 與最佳解（MILP）、自創 heuristic、Top-K 基準
   做 optimality gap 比較，以及 runtime scaling
3. **可重現性** — 任何人拿到 repo + seeds 就能重生出一模一樣的實例集

## 1. 資料夾結構

頂層 `synthetic_data/` 與 `data/` 平行，**不污染 `data/processed/`**。

```
synthetic_data/
├── PLAN.md                        # ← 本文件
├── README.md                      # 給 reader 的快速上手
├── generator/                     # 生成器（純函數，不做 I/O）
│   ├── __init__.py
│   ├── factors.py                 # DoE：InstanceSpec 27 筆
│   ├── universe.py                # stock_id / industry / market_cap / theme
│   ├── parameters.py              # μ / β / liquidity / initial_state（合併 3 → 1）
│   ├── prices.py                  # factor-model GBM
│   ├── returns_scenarios.py       # Gaussian / t / block-bootstrap
│   ├── writers.py                 # 寫 §7.1–§7.4 parquets + stage2_config.yaml
│   └── rng.py                     # SeedSequence.spawn() 散播
├── validators/
│   ├── __init__.py
│   ├── schema.py                  # parquet 欄位/dtype vs §7 契約
│   ├── economic.py                # price>0, β∈[-1,3], μ z-scored, ADV>0, 相關矩陣
│   └── feasibility.py             # Stage 1 跑通分類：OK / RELAXED / TIMEOUT / FAIL
├── runners/
│   ├── __init__.py
│   ├── generate_all.py            # CLI：吃 spec → 寫 instances/
│   ├── validate_all.py            # CLI：跑 3 層驗證 → reports/validation_report.json
│   └── benchmark_all.py           # CLI：MILP+heuristic+TopK 平行化跑 → reports/manifest.csv
├── tests/                         # pytest 單元測試（一檔對一個生成模組）
│   ├── test_factors.py
│   ├── test_universe.py
│   ├── test_parameters.py
│   ├── test_prices.py
│   ├── test_returns_scenarios.py
│   ├── test_writers.py
│   └── test_determinism.py        # 同 seed → 同 parquet bytes
├── instances/                     # 生成輸出（.gitignore）
│   └── SYN_001_smallI_lowOmega_gaussian/
│       ├── prices.parquet         # §7.1
│       ├── candidates.parquet     # §7.2
│       ├── params.parquet         # §7.3（含 mu_override 欄位）
│       ├── scenarios.parquet      # §7.4
│       ├── instance_spec.json     # factor levels + seed
│       ├── stage1_config.yaml     # 該實例專屬 Stage 1 config
│       └── stage2_config.yaml     # 該實例專屬 Stage 2 config（解決 path resolution）
├── reports/                       # benchmark / validation 結果（committed）
│   ├── manifest.csv               # 單一 source of truth
│   ├── validation_report.json
│   └── benchmark_summary.csv
└── seeds/
    └── seeds.json                 # {instance_id: seed_int}
```

並於 repo root 新增：
- `tests/test_real_pipeline_smoke.py` — 跑真實 pipeline，hash 比對 ideal_portfolio.json，
  確保新增 `synthetic_data/` 沒有意外破壞既有路徑。

## 2. DoE — Factors × Levels × Scenarios（保留 27 個實例）

### 2.1 因子與水準

| # | Factor | Levels | 備註 |
|---|---|---|---|
| F1 | Universe size \|I\| | 20 / 50 / 100 | Stress combinatorial of `z` |
| F2 | Scenario count \|Ω\| | 100 / 250 / 500 | Stress CVaR LP block |
| F3 | Industry shape | (a) 3 balanced / (b) 5 balanced / (c) 5 concentrated 60-20-10-5-5 | (c) 觸發產業上限 binding |
| F4 | Return distribution | Gaussian / t(df=5) / block-bootstrap(5d) | t 拉大 tail loss → CVaR 壓力 |
| F5 | Correlation structure | Independent / Block(intra ρ=0.5, inter ρ=0.1) / Factor(single market) | |
| F6 | β distribution | Tight N(1.0, 0.15) / Wide U[0.3, 1.8] | Wide 觸發 `beta_max` |
| F7 | μ–β correlation | None / +0.5 / -0.5 | Inverse 強迫模型放棄高 μ 換低 β |
| F8 | Price level | Uniform[20,200] / LogNormal(med 80, σ=0.8) | LogNormal 觸發 `U_i=0` lot-infeasibility |
| F9 | Initial holding | Cold (w0≡0) / Warm | Cold 觸發 τ auto-bump |
| F10 | Stress combo | Normal / Tight CVaR(0.02) / Tight ind(0.20) / Infeas trap | 最後三種觸發 relaxation fallback |

### 2.2 27 個實例（5 個 sub-set）

| Sub-set | 數量 | 目的 | 變因 |
|---|---|---|---|
| S1 — Size scaling | 9 | runtime vs (\|I\|, \|Ω\|) | F1 × F2 = 3×3 |
| S2 — Distribution stress | 6 | tail / corr 效應 | F4 × F5（排除已是 baseline 的 G×I） |
| S3 — Constraint stress | 5 | β / μ / industry binding | 每格只變一個：F6=Wide / F7=+0.5 / F7=-0.5 / F3=c / F8=LogN |
| S4 — Stress combos | 4 | relaxation fallback path | F10 × (\|I\|=50, \|Ω\|=250, F4=T, F6=Wide) |
| S5 — Warm-start | 3 | turnover binding | S1 三格（小/中/大）翻 F9=Warm |

**Baseline（中心格）**：`\|I\|=50, \|Ω\|=250, 5 balanced ind, Gaussian, Block corr,
Tight β, no μ-β corr, Uniform price, Cold start, Normal caps`。各 sub-set 都以此為原點
做 OFAT（one-factor-at-a-time）變化，便於在報告中 row-by-row 解釋。

### 2.3 S4 infeasibility trap 構造規則【實作備忘】

`model.py` 對 `min_L_sum > 1.0`、`max_U_sum < 1.0`、`selectable.sum() < K` 會 **raise
ValueError**（結構性不可行，不會走 relaxation fallback）。S4 的 infeasibility trap
**必須**透過 CVaR / turnover / β / industry 4 條限制觸發，不能讓上述結構性條件失敗。

實作上：S4 的 4 個 cell 必須保證
- `U_per_stock` 有至少 `K` 個 > 0
- `K · min(L_per_stock) ≤ 1`
- `K · max(U_per_stock) ≥ 1`

否則生成器報錯。

## 3. 參數生成規格

### 3.1 stock_id / industry / market_cap → `candidates.parquet`

- `stock_id`: `SYN0001 … SYN<|I|>`（與真實 4 位數股號區分）
- `industry`: 依 F3 多項式分佈
- `market_cap`: log-normal `exp(N(24, 1.0))`，中位 2.6×10¹⁰ TWD，>95% 過 `min_market_cap=5e9`
- `theme_tag`: 常數 `"synthetic"`

### 3.2 prices → `prices.parquet`

**Factor-model GBM**：~300 trading days，最後一日對齊 `as_of="2026-04-30"`。

```
log_return_market[t] ~ 依 F4 分佈（Gauss / scaled-t / block-bootstrap）
                       annual μ=0.06, σ=0.18
log_return_industry[j,t] ~ N(0, σ_ind)  # 只有 F5=Block 時加
log_return_idio[i,t]     ~ N(0, σ_idio_i)
log_return_i[t] = drift_i + β_i·log_return_market + indlat·industry + log_return_idio
close[i,t] = P0_i · exp(cumsum(log_return_i))
```

- 日期欄寫為真正 `datetime64[ns]`（**不**走 clean.py 的 YYYYMMDD-as-int 陷阱）
- 對 F5=Independent：在價格模擬時把 β 設 0，但 `params.beta_3m` 仍存目標 β
- 對 F4=Bootstrap：先模擬 5 年 Gaussian path → 5-day block resample → 重建 cumsum

### 3.3 scenarios → `scenarios.parquet`

兩條路徑都寫：
1. 從 prices 反推（與 `_build_scenarios` 一致）
2. 直接從 F4 抽 shape `(|Ω|, |I|)` 並 attach scenario_id

> 為什麼兩條都寫：§7.4 契約承諾 `scenarios.parquet` 存在，但 Stage 1 `data_prep`
> 也會從 prices 重算。兩條一致才安全。

### 3.4 β / μ — `params.parquet`

**β 兩階段**：先抽 target β，再餵進價格模擬器當 loading，回歸結果 ≈ target β by
construction。Clip 到 [-1, 3]。

**μ 經 Cholesky 耦合**：
```
z_β = standardize(β)
z_μ_raw ~ N(0, 1)
z_μ = ρ_μβ · z_β + sqrt(1 - ρ_μβ²) · z_μ_raw
mu_override = z_μ
```
寫到 `params.parquet` 的新欄 `mu_override`。Runner wrapper 注入。

### 3.5 liquidity — `amount`/`volume`

```
target_liq_cap_i ~ U[2L, U]                    # 多數實例
ADV20_i = target_liq_cap_i · V0 / ρ            # ρ=0.10
amount_per_day = ADV20_i · (1 + ε)             # ε ~ N(0, 0.10)
volume_per_day = int(amount_per_day / close)
```
S4 stress 時刻意把 10–20% 股票的 `target_liq_cap_i < L` 來觸發 liquidity binding。

### 3.6 initial state — `stage_overrides.yaml`

- Cold: `w0={}`, `initial_holdings={}`
- Warm: 從 top-μ 取 K' 檔，Dirichlet(α=2) 權重 → `x0_lots = round(w · V0 / P_lot)`

### 3.7 `params.parquet` 欄位

| 欄位 | 來源 |
|---|---|
| date, stock_id, close | 自 prices echo |
| market_cap | per-stock 常數，daily repeat |
| beta_3m | per-stock 常數 = target β |
| roe | `N(0.10, 0.05)` per stock |
| revenue_growth | `N(0.05, 0.10)` per stock |
| **mu_override** | 新欄：直接控制 μ（Stage 1 預設不讀，runner wrapper 注入） |

## 4. RNG 隔離（決策 #3）

```python
# generator/rng.py
import numpy as np

MODULES = ("universe", "parameters", "prices", "scenarios", "initial", "writers")

def spawn_rngs(seed: int) -> dict[str, np.random.Generator]:
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(MODULES))
    return {name: np.random.default_rng(c) for name, c in zip(MODULES, children)}
```

每個 generator 模組吃 `rng=...`，**絕不**共用一條 RNG。重構任何一個模組不會破壞其他模組
的 reproducibility。

## 5. Stage 2 config（決策 #2）

每個實例生成時，writer 一併寫 `stage2_config.yaml`，內含**絕對路徑**到該實例的
`ideal_portfolio.json` + `prices.parquet`。`stage2/data_prep.py` 用 `cfg_path.parents[2]`
做 base，但因為我們的 cfg path 是絕對的，relative resolution 仍然 OK；用絕對路徑寫
input 區段最穩。

範例 `stage2_config.yaml`：
```yaml
input:
  ideal_portfolio: /abs/path/synthetic_data/instances/SYN_001/ideal_portfolio.json
  prices:         /abs/path/synthetic_data/instances/SYN_001/prices.parquet
portfolio:
  C0: 1.0e+7
  initial_holdings: {}    # 或 warm-start dict
stage2:
  c_buy:  0.001425
  c_sell: 0.004425
  lambda_trade: 0.0005
  L: 0.01
  U: 0.10
solver:
  backend: "cbc"
  time_limit: 60
  mip_gap: 0.005
  verbose: false
```

## 6. 4 層驗證

### 6.1 Schema
parquet 欄位/dtype/index 一致性對 §7.1–§7.4。

### 6.2 Economic sanity
- `close > 0`、key 欄位無 NaN
- `beta_3m ∈ [-1, 3]`
- z-score(μ) mean≈0, std≈1
- 至少 95% 過 `min_market_cap`
- 實現相關矩陣 vs F5 規格的 Frobenius 距離 < tol

### 6.3 Feasibility
跑 Stage 1，分類：

| 狀態 | 條件 | 預期出現於 |
|---|---|---|
| OK_FEASIBLE | `Optimal` + `relaxed=False` | S1/S2/S3/S5/S4-Normal |
| OK_RELAXED  | `Optimal` + `relaxed=True` + log 非空 | S4 三個 stress cell |
| TIMEOUT_PARTIAL | `Not Solved` + objective 非 None | （非預期，flag） |
| FAIL | raise / structural infeasible | 不應出現 |

`TIMEOUT_PARTIAL` 是新分類（決策後備忘 #2）。`model.py:252` 接受 `Not Solved` 但有目標值
的情況，驗證器要把這個區分出來，否則 wall-clock 不夠時會被當 OK。

### 6.4 Stage 1 ↔ Stage 2 coherence
寫出 `ideal_portfolio.json` → 跑 Stage 2 → assert：
- `n_selected_kept ≈ K`（允許 `forced_off_high_price` 降）
- `cash_after ≥ 0`
- F8=LogNormal cell：`n_lot_infeasible > 0` AND 那些股票不出現在 order_sheet（程式化 assert）

## 7. Tests（決策 #5、#6）

### 7.1 `synthetic_data/tests/`

每檔 pytest 對一個生成器模組做 **statistical assertion**（n=200 stocks 跑 1 次以低變異）：

| 檔 | 斷言 |
|---|---|
| test_factors.py | `build_instance_specs()` 回傳剛好 27 筆，每筆 factor levels 合法 |
| test_universe.py | balanced→每 ind ≥ 1；concentrated→60-20-10-5-5 ±10% |
| test_parameters.py | corr(μ, β) ≈ ρ_μβ ± 0.1；β ∈ [-1, 3]；ADV ≥ 0 |
| test_prices.py | close>0；regression β ≈ target β ± 0.15；date dtype = datetime64[ns] |
| test_returns_scenarios.py | t-分佈 kurtosis > 3.5；block corr matrix Frobenius < tol |
| test_writers.py | 寫出 4 個 parquet + 2 個 yaml/json；reload via `build_stage1_inputs` 成功 |
| test_determinism.py | 同 seed 跑兩次 → parquet bytes 相同 |

### 7.2 `tests/test_real_pipeline_smoke.py`（repo root）

跑既有 `MILP_phase_1.stage1.solve` + `MILP_phase_2.stage2.solve`，hash 比對
`ideal_portfolio.json` 的 weights dict。確保新增 `synthetic_data/` 沒有意外影響真實
pipeline。

## 8. Benchmark（決策 #7、#8）

`runners/benchmark_all.py` 用 `multiprocessing.Pool` 平行跑 4 method × 27 instance：

| Method | 設定 |
|---|---|
| MILP optimal | mip_gap=0.001 (CBC) or 0.0 (Gurobi)，time_limit=300s |
| MILP default | mip_gap=0.005, time_limit=120s |
| Greedy heuristic | `MILP_phase_1.stage1.heuristic.greedy_heuristic` |
| Top-K equal | `MILP_phase_1.stage1.heuristic.top_k_equal_weight` |

寫 `reports/manifest.csv`，欄位涵蓋全部 10 個 factor levels + 每 method 的
`objective / runtime / status / gap_pct`。

### 8.1 Wall-clock 預期【實作備忘 #3】

- CBC、serial：3–4 hours
- CBC、parallel(4 cores)：~1 hour
- Gurobi：<30 min（無論平行）

## 9. Git tracking（決策 #4）

`.gitignore` 加：
```
synthetic_data/instances/
```

Commit：generator/、validators/、runners/、tests/、reports/、seeds/、PLAN.md、README.md。

任何人 clone repo + `python -m synthetic_data.runners.generate_all` 即可重生出
bit-for-bit 相同的 27 個實例。

## 10. 明確的非目標

- ❌ 不動 `data/processed/`
- ❌ 不動 `MILP_phase_1/stage1/*.py`
- ❌ 不動 `MILP_phase_2/stage2/*.py`（除非加 `--processed-dir` 之類，但本計畫繞過了）
- ❌ 不動 §7 介面契約（`mu_override` 是 additive 新欄）
- ❌ 不取代真實實例
- ❌ 不引入新外部依賴（numpy / pandas / pyarrow / pyyaml / scipy.stats 已在 requirements）
- ❌ 不做 multi-period rolling backtest（Module 5 / D 的職責）
- ❌ 不出 binary / package
- ❌ 不做 Fama-French 多因子（單因子模型對課程足夠）
- ❌ 不用 Plackett-Burman 等真 fractional factorial（OFAT 更易報告解釋）

## 11. 執行順序

1. **Phase A**（scaffold）：folder tree, `.gitignore`, `__init__.py`, `rng.py`, `factors.py`
2. **Phase B**（generators + tests）：parameters / prices / scenarios / writers + 對應 test
3. **Phase B**（runner）：`generate_all.py`
4. **Phase C**（validators + runner）：3 validators + `validate_all.py`
5. **Phase E-pre**（smoke）：生 1 個小實例 → 完整 validate + benchmark
6. **Phase D**（benchmark）：`benchmark_all.py`（multiprocessing.Pool）
7. **Phase E**（reports）：跑全 27 個實例、產 manifest.csv、寫 README.md

預估時數（with AI assist）：scaffold 0.5h、generators 2h、tests 1.5h、validators 1h、
runners 1h、benchmark sweep 1h（平行）= ~7h 總開發 + ~1h benchmark wall-clock。

---

## Review log

| # | 決策 | 來源 | 結論 |
|---|---|---|---|
| 1 | 實例數 | AskUserQuestion | 保留 27 |
| 2 | Stage 2 config | AskUserQuestion | 每實例自帶 `stage2_config.yaml` |
| 3 | RNG 隔離 | AskUserQuestion | `SeedSequence.spawn()` |
| 4 | Git tracking | AskUserQuestion | gitignore instances/ |
| 5 | 模組合併 | AskUserQuestion | `parameters.py` 合併 3 個 |
| 6 | 生成器單測 | AskUserQuestion | 每模組一個 pytest |
| 7 | 回歸測試 | AskUserQuestion | `tests/test_real_pipeline_smoke.py` |
| 8 | 平行化 | AskUserQuestion | `multiprocessing.Pool` |
| 9 | S4 trap 構造規則 | 實作備忘 | 必須走 CVaR/τ/β/ind, **不能** trip 結構性 ValueError |
| 10 | TIMEOUT_PARTIAL 分類 | 實作備忘 | feasibility 驗證器新增第 3 種狀態 |
| 11 | Wall-clock 文件化 | 實作備忘 | CBC serial 3–4h / parallel 1h / Gurobi <30min |

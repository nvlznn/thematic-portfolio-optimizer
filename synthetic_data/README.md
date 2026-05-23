# 合成測試實例組（Synthetic Instance Suite）

> 滿足課程要求 PDF §5「**Including both self-generated instances and at least
> one real-world instance is a must**」。對應 27 個結構化合成實例 + Stage 1/2
> MILP 求解 + 自創 heuristic + Top-K 基準的對照表。

完整設計請見 [`PLAN.md`](./PLAN.md)（含 11 項 review 決策）。

---

## 三條 CLI

從專案根目錄執行：

```bash
# 1. 從 seeds 生 27 個實例（每次跑會覆蓋 instances/）
.venv/bin/python -m synthetic_data.runners.generate_all

# 2. 跑 schema → economic → feasibility 三層驗證
.venv/bin/python -m synthetic_data.runners.validate_all

# 3. 跑 4-method benchmark：MILP optimal / MILP default / heuristic / Top-K
.venv/bin/python -m synthetic_data.runners.benchmark_all --workers 4
```

每條命令都支援 `--only SYN_001 SYN_005` 來只處理指定實例。

## 27 個實例的 DoE

| 子集 | 數量 | 變因 | 目的 |
|---|---|---|---|
| **S1 — 規模掃描** | 9 | \|I\| × \|Ω\| = {20, 50, 100} × {100, 250, 500} | runtime vs 問題規模 |
| **S2 — 分佈壓力** | 6 | F4 × F5 ∈ {t, bootstrap} × {independent, block, factor} | tail / 相關結構 |
| **S3 — 約束壓力** | 5 | wide β / μ-β=+0.5 / μ-β=-0.5 / 集中產業 / log-normal 價格 | 單一限制 binding |
| **S4 — 不可行陷阱** | 4 | normal / tight CVaR(0.008) / tight industry(0.15) / infeas trap | relaxation fallback 路徑 |
| **S5 — Warm-start** | 3 | S1 對角三格翻 F9=warm | turnover 限制 binding |

10 個因子的完整 level 表見 `PLAN.md` §2.1。所有非變因都鎖在 baseline
`|I|=50, |Ω|=250, 5 balanced industries, gaussian, block correlation, tight β,
no μ-β corr, uniform price, cold start, normal caps`。

## 驗證結果（最新一次跑）

```
分類匯總：{'OK_FEASIBLE': 24, 'OK_RELAXED': 3}
```

3 個 OK_RELAXED 全部是 S4 stress instances，relaxation_log 完整記錄了
`cvar → turnover → beta → industry` 的順序：

```
SYN_022_stress_tight_cvar       log: [Infeasible, 放寬限制：cvar]
SYN_023_stress_tight_industry   log: [Infeasible, cvar, Infeasible, turnover, Infeasible, beta, Infeasible, industry]
SYN_024_stress_infeas_trap      log: [Infeasible, cvar, Infeasible, turnover, Infeasible, beta, Infeasible, industry]
```

這就是報告 §6 需要的「relaxation 機制實際運作」的證據。

## Benchmark 摘要（4 方法 × 27 實例）

平均 optimality gap（% vs MILP optimal）：

| 子集 | n | mean opt obj | mean heuristic gap | mean Top-K gap |
|---|---|---|---|---|
| S1 — 規模掃描 | 9 | 1.18 | 0.00% | 0.42% |
| S2 — 分佈壓力 | 6 | 1.32 | 0.00% | 0.28% |
| S3 — 約束壓力 | 5 | 1.21 | 0.00% | 1.61% |
| S4 — 不可行陷阱 | 4 | 1.30 | 31.87% | 31.97% |
| S5 — Warm-start | 3 | 1.14 | -2.50% | -4.30% |

**讀法**：
- S1/S2/S3 — 在常規可行的實例上，heuristic 幾乎完全匹配 MILP，Top-K 落後 0.3–1.6%。
- S4 — 嚴重 stress 下 heuristic 大幅落後（因為 MILP 透過 relax 找到了原始模型外的解）；
  這對比凸顯了「自創 heuristic 受限於原始約束、無 relaxation」的 trade-off。
- S5 — warm-start 場景中 heuristic 略勝 MILP；MILP 的 turnover 限制在某些 seed
  下未綁定，heuristic 反而在較自由的空間裡找到較高目標。值得在報告 §6 對應一段討論。

Heuristic 擊敗 Top-K：**26/27**。唯一例外 `SYN_025_warm_20x100` 是 small universe
+ warm-start 的邊角案例。

完整 manifest（每個實例 35 欄）：[`reports/manifest.csv`](./reports/manifest.csv)。

## 檔案結構

```
synthetic_data/
├── PLAN.md                # 完整設計 + 11 項 review 決策
├── README.md              # ← 你正在看
├── generator/             # 純函數生成器（不做 I/O）
│   ├── factors.py         # 27-row DoE 矩陣
│   ├── universe.py        # stock_id / industry / market_cap
│   ├── parameters.py      # β / μ / liquidity / initial state（合併三模組）
│   ├── prices.py          # factor-model GBM with F4/F5 branching
│   ├── returns_scenarios.py
│   ├── writers.py         # 寫 4 parquets + 2 yaml/json + ideal_portfolio.json
│   └── rng.py             # SeedSequence.spawn() per-module RNG isolation
├── validators/
│   ├── schema.py          # §7.1–§7.4 欄位/dtype 一致性
│   ├── economic.py        # price>0 / β∈[-1,3] / 相關矩陣 sanity
│   └── feasibility.py     # Stage 1+2 求解分類 + mu_override 注入
├── runners/
│   ├── generate_all.py    # CLI：spec → instances/
│   ├── validate_all.py    # CLI：3 層驗證
│   └── benchmark_all.py   # CLI：multiprocessing 4-method sweep
├── tests/                 # pytest 25 個單元測試
│   ├── test_factors.py
│   ├── test_universe.py
│   ├── test_parameters.py
│   ├── test_prices.py
│   └── test_determinism.py
├── instances/             # .gitignore — 從 seeds 重生
│   └── SYN_NNN_*/
│       ├── prices.parquet     # §7.1
│       ├── candidates.parquet # §7.2
│       ├── params.parquet     # §7.3（含新欄 mu_override）
│       ├── scenarios.parquet  # §7.4
│       ├── instance_spec.json
│       ├── stage1_config.yaml
│       └── stage2_config.yaml
├── reports/
│   ├── manifest.csv           # benchmark 主表（35 欄 × 27 列）
│   └── validation_report.json
└── seeds/
    └── seeds.json             # {instance_id: seed_int}
```

## 重現性

`pytest synthetic_data/tests/test_determinism.py` 驗證：同 seed → 兩次跑出的
4 個 parquet bytes 完全相同。

```bash
.venv/bin/python -m pytest synthetic_data/tests/ -v
# 25 passed in ~17s
```

## 與真實 pipeline 的關係

合成實例是真實實例的**互補**，不是替代：

- 真實實例（`data/processed/202604/`）：1 個，台股 2026-04，3461 → ~80 stock
  universe。展示 practicality。
- 合成實例（`synthetic_data/instances/`）：27 個，跨 10 個因子的結構化變化。
  展示 algorithm applicability + relaxation fallback evidence + heuristic
  optimality gap。

兩者用同一套 Stage 1/2 程式碼，不修改任何 `MILP_phase_1/` 或 `MILP_phase_2/`
裡的檔案。合成實例的 `params.parquet` 多一個 `mu_override` 欄（additive
extension），由 `validators/feasibility.py` 與 `runners/benchmark_all.py` 注入
到 `Stage1Inputs.mu` — F7（μ-β 相關）這樣才會真正生效。

回歸測試在 repo root 的 `tests/test_real_pipeline_smoke.py`，確保新增
`synthetic_data/` 沒有破壞真實 pipeline。

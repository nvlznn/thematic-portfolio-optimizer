# 第一階段 MILP — 理想連續權重組合

> 對應 [`開發計畫.md`](../開發計畫.md) Module 3、[`proposal.pdf`](../proposal.pdf) §6.2，
> 介面契約 §7.3（`params.parquet`）→ §7.5（`ideal_portfolio.json`）。

本資料夾是 **C — 第一階段 MILP** 的程式碼與輸出。輸入是 A、B 已產出的 parquet
介面契約檔，輸出是 Stage 2 直接吃的 `ideal_portfolio.json`。

---

## 1. 目錄結構

```
MILP_phase_1/
├── README.md                ← 本文件
├── stage1/
│   ├── __init__.py
│   ├── config.yaml          ← 所有可調參數（K、L、U、γ、β^max、τ、α、C̄^risk 等）
│   ├── data_prep.py         ← 把 §7.1/§7.2/§7.3 parquet 轉成 MILP 輸入
│   ├── model.py             ← MILP 建模 + 求解 + relax fallback
│   └── solve.py             ← CLI 入口
└── output/
    └── ideal_portfolio.json ← §7.5 介面契約輸出，餵給 Stage 2
```

---

## 2. 環境準備

### 2.1 建立虛擬環境

```bash
# 在專案根目錄
python3.13 -m venv .venv          # 或 python3.12 / 3.11
source .venv/bin/activate
pip install --upgrade pip
pip install pyarrow pandas numpy pulp pyyaml
```

> 完整套件清單見專案根目錄 [`requirements.txt`](../requirements.txt)，
> Stage 1 只需要上面 5 個。

### 2.2 求解器

預設使用 PuLP 內建的 **CBC**（隨 `pulp` 一起裝好，無需額外設定）。
若有 Gurobi 學術授權，把 `stage1/config.yaml` 的 `solver.backend` 改為 `gurobi`
或 `auto`，程式會優先呼叫 Gurobi。

---

## 3. 執行

從專案根目錄執行：

```bash
.venv/bin/python -m MILP_phase_1.stage1.solve
```

預設參數：

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--config` | `MILP_phase_1/stage1/config.yaml` | 設定檔位置 |
| `--processed` | `data/processed` | 介面契約 parquet 來源 |
| `--output` | `MILP_phase_1/output/ideal_portfolio.json` | §7.5 輸出位置 |

執行範例輸出：

```
[Stage1] 讀取資料： data/processed
[Stage1] 候選股 1046 檔、產業 34 個、情境 240 筆、rebalance_date=2026-04-30
[Stage1] 求解中...
[Stage1] 狀態=Optimal、目標=2.3534、持股=20、β=1.100、CVaR=0.0400、turnover=1.000
[Stage1] 已輸出 MILP_phase_1/output/ideal_portfolio.json
```

---

## 4. 模型細節

完整公式見 `proposal.pdf` §6.2 與 `開發計畫.md` §A。重點摘要：

* **目標**：$\max \sum_i \mu_i w_i$
* **變數**：$w_i \ge 0, z_i \in \{0,1\}, \delta^+_i, \delta^-_i \ge 0, \eta, \xi_s \ge 0$
* **限制**：
  1. 預算 $\sum w_i = 1$
  2. 持股檔數 $\sum z_i = K$
  3. 持股權重上下限 $L z_i \le w_i \le U z_i$
  4. 產業集中度 $\sum_{i \in S_j} w_i \le \gamma_j$
  5. Beta 上限 $\sum \beta_i w_i \le \beta^{\max}$
  6. 流動性 $w_i \le \ell_i$
  7. 周轉率（線性化）$\sum (\delta^+_i + \delta^-_i) \le \tau$
  8. CVaR（Rockafellar–Uryasev 線性化）$\eta + \frac{1}{(1-\alpha)|\Omega|}\sum \xi_s \le \bar{C}^{\text{risk}}$

### 4.1 首期 (`w0 = 0`) 自動處理

若期初權重全為零，從零部位起的最小周轉率必為 $\sum w_i = 1$，因此程式內部會
自動把 $\tau$ 拉到 $2.0$（最大可能值）以避免結構性不可行。

### 4.2 不可行 fallback

若 MILP 不可行，`model.py` 會依序放寬：`cvar → turnover → beta → industry`，
並在 `diagnostics.relaxation_log` 紀錄哪些限制被放寬，供下游診斷。

### 4.3 後驗 CVaR

`diagnostics.cvar_realised` 是 **用最終 $w$ 直接在情境上重新計算**的 CVaR，
不是讀 $\eta, \xi$ 變數值（因為被 relax 時那些變數會飄移）。

---

## 5. 介面契約

### 5.1 輸入 — `data/processed/`

| 檔名 | 來源 | 主要欄位 |
|---|---|---|
| `prices.parquet` | A | `date, stock_id, close, volume, amount` |
| `candidates.parquet` | A | `stock_id, name, market_cap, industry, theme_tag` |
| `params.parquet` | B | `date, stock_id, close, beta_3m, roe, revenue_growth, ...` |

> 目前 `clean.py` 的 `date` 欄是把 `YYYYMMDD` 直接餵給 `pd.to_datetime`
> 而被解讀為 ns since epoch；`data_prep._decode_dates` 會偵測並還原。
> A 修好之後本端會自動相容。

### 5.2 輸出 — `output/ideal_portfolio.json`

完全遵循 §7.5：

```json
{
  "as_of": "2026-04-30",
  "objective_value": 2.3534,
  "weights": {"2330": 0.10, "...": 0.0},
  "selected": ["2330", "..."],
  "diagnostics": {
    "status": "Optimal",
    "relaxed": false,
    "relaxation_log": [],
    "active_constraints": ["beta_max", "cvar_max"],
    "cvar_realised": 0.0400,
    "turnover_realised": 1.000,
    "beta_realised": 1.100,
    "industry_weights": {"半導體業": 0.39, ...},
    "input_summary": {...},
    "config": {...}
  }
}
```

---

## 6. 參數調整速查

`stage1/config.yaml` 常用旋鈕：

| 區段 | 參數 | 影響 |
|---|---|---|
| `universe.min_market_cap` | `5e9` | 提高 → 候選股變少、流動性更好 |
| `mu.method` | `momentum`/`blend` | `blend` 會混入 ROE z-score |
| `liquidity.rho` | `0.1` | 單檔部位佔 ADV 的最大比例 |
| `stage1.K` | `20` | 目標持股檔數 |
| `stage1.U` | `0.10` | 單檔上限（必須 ≥ 1/K） |
| `stage1.beta_max` | `1.10` | 越低越保守 |
| `stage1.turnover_max` | `0.50` | 首期會自動忽略 |
| `stage1.cvar_max` | `0.04` | 越低越保守，太低會 infeasible |
| `stage1.industry_caps` | dict | 每個產業的單獨權重上限 |
| `solver.backend` | `cbc`/`gurobi`/`auto` | 求解器 |

---

## 7. 與下游模組（Stage 2）的銜接

D 寫的 Stage 2 從本輸出讀取：

* `weights` → $w_i^\star$（理想連續權重）
* `selected` → $z_i^\star$（持股二元決策）
* `as_of` → 對齊 `prices.parquet` 的當期報價 $P_i$

任何 schema 改動請按開發計畫 §7 先發 PR 討論。

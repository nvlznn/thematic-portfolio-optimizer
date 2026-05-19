# Thematic Portfolio Optimizer

> NTU 114-2 作業研究期末專案：**主題型股票投資之兩階段再平衡決策支援系統**
>
> 結合投資組合最佳化、整數張數轉換與現金流守恆之混合整數規劃模型。

[proposal.pdf](proposal.pdf) · [開發計畫.md](開發計畫.md) · [Stage 1 README](MILP_phase_1/README.md) · [Stage 2 README](MILP_phase_2/README.md)

---

## 系統概覽

把人工執行的「主題股投資再平衡」流程，拆成 5 個自動化模組、用 MILP 一氣呵成解出 **可實際下單** 的調倉清單：

```
data/raw/YYYYMM/*.csv                            ← TEJ 原始下載
        ▼  src/data/clean.py
data/processed/{prices,candidates,params}.parquet ← §7.1–§7.3 介面契約
        ▼  MILP_phase_1/  (Module 3 — Stage 1 MILP)
MILP_phase_1/output/ideal_portfolio.json         ← §7.5（w*, z*, diagnostics）
        ▼  MILP_phase_2/  (Module 4 — Stage 2 MILP)
MILP_phase_2/output/order_sheet.csv              ← §7.6（u, v, x, C）
        ▼  (Module 5 — 回測，TBD)
回測報告與 benchmark 對比
```

- **Stage 1**：在報酬潛力、風險控制（Beta、CVaR）、產業集中度、流動性、周轉率限制下，求出理想 **連續權重** $w_i^\star$ 與持股二元決策 $z_i^\star$。
- **Stage 2**：把理想權重落地為滿足整數張數、現金守恆、買賣狀態互斥的可下單清單 $u_i, v_i, x_i$。

兩階段都是 MILP，可用 Gurobi / CBC 求解；本實作預設用 PuLP + CBC（無需額外授權）。

---

## 快速開始

```bash
# 1. 環境
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 資料清理（從 data/raw/YYYYMM/*.csv → data/processed/*.parquet）
python src/data/clean.py

# 3. Stage 1：求理想權重
python -m MILP_phase_1.stage1.solve

# 4. Stage 2：轉成可下單張數
python -m MILP_phase_2.stage2.solve
```

執行範例輸出：

```
[Stage1] 候選股 1046 檔、產業 34 個、情境 240 筆、rebalance_date=2026-04-30
[Stage1] 狀態=Optimal、目標=2.0616、持股=20、β=1.100、CVaR=0.0400、turnover=1.000
[Stage2] universe 20 檔、保留 Stage1 持股 20 檔、V0=10,000,000、as_of=2026-04-30
[Stage2] 狀態=Optimal、Σ deviation=0.0177、交易標的=20、投入=9,977,350、剩餘現金=8,432（0.08%）
```

---

## 目錄結構

```
thematic-portfolio-optimizer/
├── proposal.pdf                ← 修正版 proposal（§6 完整 MILP 公式）
├── 開發計畫.md                  ← 4 人分工、§7 介面契約、時程
├── requirements.txt
├── CLAUDE.md                   ← 給 Claude Code 的 repo 上手指引
│
├── data/
│   ├── raw/202604/*.csv        ← TEJ 原始下載（A 提供）
│   └── processed/*.parquet     ← 介面契約檔（下游全部消費這個）
│
├── src/data/
│   └── clean.py                ← raw → processed 的清洗 pipeline
│
├── MILP_phase_1/               ← Stage 1：理想連續權重（C 負責）
│   ├── stage1/{config.yaml,data_prep,model,solve}.py
│   ├── output/ideal_portfolio.json
│   └── README.md
│
└── MILP_phase_2/               ← Stage 2：整數張數落地（D 負責）
    ├── stage2/{config.yaml,data_prep,model,solve}.py
    ├── output/{order_sheet.csv,stage2_diagnostics.json}
    └── README.md
```

---

## 介面契約（§7 — 跨人協作命脈）

| 檔案 | §  | 產出者 | 消費者 | 主要欄位 |
|---|---|---|---|---|
| `data/processed/prices.parquet` | 7.1 | A | Stage 1, Stage 2, 回測 | `date, stock_id, close, volume, amount` |
| `data/processed/candidates.parquet` | 7.2 | A | Stage 1 | `stock_id, name, market_cap, industry, theme_tag` |
| `data/processed/params.parquet` | 7.3 | B | Stage 1 | `date, stock_id, close, beta_3m, roe, revenue_growth` |
| `MILP_phase_1/output/ideal_portfolio.json` | 7.5 | C | Stage 2 | `{as_of, weights, selected, diagnostics}` |
| `MILP_phase_2/output/order_sheet.csv` | 7.6 | D | 回測 / 下單 | `stock_id, price, x0_lots, buy_lots, sell_lots, x_lots, weight_actual, weight_ideal, deviation, cash_after` |

> 任何 schema 改動請按開發計畫 §7 先發 PR 討論再合進來，不要單方面動。

---

## 模型重點

完整公式見 [proposal.pdf](proposal.pdf) §6，這裡只列幾個容易誤解的設計選擇：

- **per-stock L/U**：proposal §6.2 是純量 $L, U$，但 Stage 2 整數張數會把每檔的可達權重限制成 $\{k \cdot P^{lot}_i / V^0\}$ 離散集。為了讓兩階段共用同一可行域，Stage 1 自動把 $L, U$ 換成 per-stock 的 $L_i = \lceil L V^0 / P_i^{lot} \rceil \cdot P_i^{lot} / V^0$、$U_i = \lfloor U V^0 / P_i^{lot} \rfloor \cdot P_i^{lot} / V^0$。詳見 [Stage 1 README §4.1](MILP_phase_1/README.md)。
- **首期 (`w0 ≡ 0`) 自動處理**：從零部位起的最小周轉率必為 $\sum w_i = 1$，任何 $\tau < 1$ 都會結構性不可行。Stage 1 內部會自動把 $\tau$ 拉到 2.0。
- **不可行 fallback**：Stage 1 若 infeasible，依序放寬 `cvar → turnover → beta → industry`，並把放寬紀錄寫進 `diagnostics.relaxation_log`。
- **後驗 CVaR**：`cvar_realised` 直接用最終 $w$ 在情境上重新計算，不讀 LP 的 $\eta, \xi$ 變數（被 relax 時那些變數會飄移）。

---

## 與 benchmark 比較（回測時用）

依 [proposal.pdf](proposal.pdf) §7，回測會比較三個策略：

| Benchmark | 內容 |
|---|---|
| **B1**：Top-K 等權重 | 依吸引力分數選前 K 檔，平均配置 |
| **B2**：單階段連續權重 | 只解 Stage 1，不處理整數張數與現金守恆 |
| **B3**：本研究兩階段決策支援系統 | 同時處理理想配置與可執行交易清單 |

評估指標：累積/年化報酬、年化波動、Sharpe、最大回撤、平均周轉率、平均持股檔數、實際交易標的數、現金佔比、理想 vs 實際偏離。

---

## 課程資訊

- **課程**：114-2 作業研究（NTU）
- **小組**：4 人組 — A（資料）/ B（參數）/ C（Stage 1，劉威廷、闕以諾）/ D（Stage 2 + 回測）
- **時程與分工**：詳見 [開發計畫.md](開發計畫.md) §6、§8

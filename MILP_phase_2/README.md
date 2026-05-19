# 第二階段 MILP — 交易落地與整數張數轉換

> 對應 [`開發計畫.md`](../開發計畫.md) Module 4、[`proposal.pdf`](../proposal.pdf) §6.3，
> 介面契約 §7.5 (`ideal_portfolio.json`) → §7.6 (`order_sheet.csv`)。

把 Stage 1 的「理想連續權重」轉成滿足整數張數、現金守恆、買賣狀態限制的
**可實際下單** 調倉清單。

---

## 1. 目錄結構

```
MILP_phase_2/
├── README.md
├── stage2/
│   ├── __init__.py
│   ├── config.yaml          ← C0、c_buy、c_sell、λ_trade、L、U
│   ├── data_prep.py         ← 載入 Stage1 輸出 + 算 P_i、L_lot、U_lot、ū_i
│   ├── model.py             ← MILP 建模 + 求解
│   └── solve.py             ← CLI 入口
└── output/
    ├── order_sheet.csv          ← §7.6 介面契約輸出
    └── stage2_diagnostics.json  ← 總投入、剩餘現金、偏離、交易筆數等
```

---

## 2. 執行

需先跑過 [Stage 1](../MILP_phase_1/) 並產出 `MILP_phase_1/output/ideal_portfolio.json`。

```bash
# 在專案根目錄
python -m MILP_phase_2.stage2.solve
```

預設參數：

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--config` | `MILP_phase_2/stage2/config.yaml` | 設定檔位置 |
| `--output` | `MILP_phase_2/output/order_sheet.csv` | §7.6 輸出位置 |
| `--diagnostics` | `MILP_phase_2/output/stage2_diagnostics.json` | 摘要診斷檔 |

執行範例輸出：

```
[Stage2] 讀取設定： MILP_phase_2/stage2/config.yaml
[Stage2] universe 20 檔、保留 Stage1 持股 20 檔、V0=10,000,000、as_of=2026-04-30
[Stage2] 求解中...
[Stage2] 狀態=Optimal、目標=0.0274、Σ deviation=0.018、交易標的=20、
         投入=9,941,200、剩餘現金=44,617（0.45%）
[Stage2] 已輸出 MILP_phase_2/output/order_sheet.csv
```

---

## 3. 模型細節

完整公式見 `proposal.pdf` §6.3 與 `開發計畫.md` §A。重點：

* **決策變數**
  * $x_i \in \mathbb{Z}_{\ge 0}$：調倉後最終張數
  * $u_i \in \mathbb{Z}_{\ge 0}$：買入張數
  * $v_i \in \mathbb{Z}_{\ge 0}$：賣出張數
  * $C \ge 0$：調倉後現金
  * $B_i, Q_i \in \{0,1\}$：是否買入 / 賣出
  * $d_i \ge 0$：實際 vs 理想配置之絕對偏離

* **目標**：$\min \sum_i d_i + \lambda_{\text{trade}} \sum_i (B_i + Q_i)$

* **限制**
  1. 庫存守恆：$x_i = x_i^0 + u_i - v_i$
  2. 現金守恆：$C = C^0 + \sum P_i v_i(1-c_s) - \sum P_i u_i(1+c_b)$、$C \ge 0$
  3. L1 偏離線性化：$d_i \ge \pm (P_i x_i / V^0 - w_i^\star)$
  4. 持股名單連結：$L_i^{\text{lot}} z_i^\star \le x_i \le U_i^{\text{lot}} z_i^\star$
  5. 買入 / 二元連結：$u_i \le \bar{u}_i B_i$
  6. 賣出 / 二元連結：$v_i \le x_i^0 Q_i$
  7. 互斥：$B_i + Q_i \le 1$

其中：
* $V^0 = C^0 + \sum P_i x_i^0$（期初總資產，**1 張 = 1000 股**）
* $L_i^{\text{lot}} = \lceil L V^0 / P_i \rceil$，$U_i^{\text{lot}} = \lfloor U V^0 / P_i \rfloor$
* $\bar{u}_i = \max(0, U_i^{\text{lot}} - x_i^0)$

### 3.1 高股價自動處理

部分高股價股票會出現 $L_i^{\text{lot}} > U_i^{\text{lot}}$（box 空集合）甚至 $U_i^{\text{lot}} = 0$
（連 1 張都買不起 $U$ 比例的權重）：

* **自動降低 $L_i^{\text{lot}} \leftarrow U_i^{\text{lot}}$**：保留持股可能性、放棄下限
* **$U_i^{\text{lot}} = 0$ 的股票自動退出 $z^\star$**：等同 Stage 1 未選；列入診斷檔以利覆盤

實務上這對應 proposal §10 列出之模型限制 (2)：「持股下限 + 流動性 + 整數張數
條件同時過嚴時可能不可行」。

### 3.2 Universe 處理

第二階段的 universe 不只是 $z^\star = 1$ 的股票，也包含 $x_i^0 > 0$ 但已不在
$z^\star$ 內的「待清倉」股票（這時 $L^{\text{lot}} = U^{\text{lot}} = 0$，
模型會強制 $v_i = x_i^0$）。

---

## 4. 介面契約

### 4.1 輸入

| 來源 | 路徑（預設） | 用途 |
|---|---|---|
| Stage 1 輸出 | `../MILP_phase_1/output/ideal_portfolio.json` | $w_i^\star$、$z_i^\star$、`as_of` |
| §7.1 價格 | `../data/processed/prices.parquet` | 取 `as_of` 當日收盤價 |
| 本端設定 | `stage2/config.yaml` | $C^0$、$x_i^0$、$c_b$、$c_s$、$\lambda_{\text{trade}}$ |

### 4.2 輸出 — `output/order_sheet.csv`（§7.6）

| 欄位 | 型別 | 說明 |
|---|---|---|
| `stock_id` | str | 證券代碼 |
| `price` | float | $P_i$（元 / 張） |
| `x0_lots` | int | 期初張數 $x_i^0$ |
| `buy_lots` | int | 買入張數 $u_i$ |
| `sell_lots` | int | 賣出張數 $v_i$ |
| `x_lots` | int | 期末張數 $x_i$ |
| `weight_actual` | float | $P_i x_i / V^0$ |
| `weight_ideal` | float | $w_i^\star$ |
| `deviation` | float | $d_i$ |
| `cash_after` | float | 調倉後現金 $C$（每列同值，方便聚合） |

### 4.3 診斷檔 — `output/stage2_diagnostics.json`

供報表與回測讀取，包含：

* 目標值、總偏離、交易筆數、剩餘現金佔比
* 自動處理過的高股價股票清單
* 完整 Stage 2 config 副本（重現用）

---

## 5. 參數調整速查

`stage2/config.yaml` 常用旋鈕：

| 區段 | 參數 | 影響 |
|---|---|---|
| `portfolio.C0` | `1e7` | 期初現金；首期 V0 = C0 |
| `portfolio.initial_holdings` | dict | 期初持股 `{stock_id: x0_lots}`，回測下一期會帶入 |
| `stage2.c_buy` | `0.001425` | 買入手續費率 |
| `stage2.c_sell` | `0.004425` | 賣出手續費率 + 證交稅 |
| `stage2.lambda_trade` | `0.0005` | 越大越偏好少標的交易（避免零碎下單） |
| `stage2.L` / `stage2.U` | 對齊 Stage 1 | 持股最低/最高權重 |
| `solver.backend` | `cbc`/`gurobi`/`auto` | 求解器 |

---

## 6. 與下游模組（Stage 5 回測）的銜接

回測（Module 5）的 rolling rebalance loop 每一個再平衡日會：

1. 跑 Stage 1 → 產 `ideal_portfolio.json`
2. 把上一期的 `x_lots`、`cash_after` 寫進 Stage 2 config 當作 `initial_holdings` 與 `C0`
3. 跑 Stage 2 → 產 `order_sheet.csv`、更新 portfolio state
4. 用下一期實際報酬計算策略績效
5. 進入下一個 rebalance 日

---

## 7. 與 Stage 1 一起跑

```bash
# 在專案根目錄一次跑完
python -m MILP_phase_1.stage1.solve && python -m MILP_phase_2.stage2.solve
```

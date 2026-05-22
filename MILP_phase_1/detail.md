# Stage 1 實作細節

## 介面契約(§7,跨模組勿單方面改)
| 檔案 | 角色 | 主要欄位 |
|---|---|---|
| `data/processed/prices.parquet` | 輸入(A) | date, stock_id, close, volume, amount |
| `data/processed/candidates.parquet` | 輸入(A) | stock_id, name, market_cap, industry, theme_tag |
| `data/processed/params.parquet` | 輸入(B) | date, stock_id, close, beta_3m, roe, revenue_growth |
| `output/ideal_portfolio.json` | 輸出 → Stage 2 | as_of, weights, selected, diagnostics |

## 設計選擇(非顯而易見)
- **per-stock L/U**：proposal 是純量 L、U,但 Stage 2 整數張數讓每檔可達權重是離散格點。為了兩階段共用同一可行域,Stage 1 自動換成 `L_i=⌈L·V0/P_lot⌉·P_lot/V0`、`U_i=⌊U·V0/P_lot⌋·P_lot/V0`;當 `U_i=0`(1 張就超過 U·V0)強制 `z_i=0`。算在 `data_prep._build_lot_bounds`。
- **首期 (w0≡0)**：從零部位起的最小周轉率 = Σw = 1,任何 τ<1 結構性不可行 → model 自動把 τ 拉到 2.0,別在 config 調低。
- **後驗 CVaR**：`diagnostics.cvar_realised` 用最終 w 在情境上重算,不讀 LP 的 η/ξ(被 relax 時會飄)。
- **不可行 fallback**：infeasible 時依序放寬 `cvar → turnover → beta → industry`,記在 `relaxation_log`。
- **1 張 = 1000 股**：`LOT_SIZE = 1000`。

## 求解器
- 預設 PuLP + **CBC**(開源、無 size 限制,可跑全 universe ~1046 檔)。`solver.backend: gurobi|auto` 才試 Gurobi。
- pip 版 Gurobi 是 restricted license:**上限約 2000 變數/限制式 ≈ 430 檔**,全 universe 會超過 → 用 CBC 或申請台大學術授權(無限制)。
- `mip_gap`：預設 `0.005`(0.5% 容差,**非真最佳**)。要當 benchmark 的最佳解上界,設 `0` 並放大 `time_limit`;`benchmark.py` 已自動設 0。

## 資料 gotcha
- **日期編碼**：`clean.py` 把 YYYYMMDD 直接餵 `to_datetime` 被當成 ns since epoch(變 1970 年);`data_prep._decode_dates` 會偵測並還原,上游修好後自動相容。
- **YAML 科學記號**：PyYAML 把 `5.0e9` 當字串,要寫 `5.0e+9`(明確正號)。

## config.yaml 速查
| 參數 | 影響 |
|---|---|
| `universe.min_market_cap` | 提高 → 候選股變少、流動性更好 |
| `stage1.K` | 目標持股檔數 |
| `stage1.L` / `U` | 單檔權重下 / 上限(會 round 到整張倍數) |
| `stage1.beta_max` / `cvar_max` | 越低越保守(太低會 infeasible) |
| `stage1.turnover_max` | 首期自動忽略 |
| `stage1.industry_caps` | 每產業權重上限 |
| `solver.backend` / `mip_gap` | cbc/gurobi/auto、最佳化容差 |

## benchmark 用法
```bash
python -m MILP_phase_1.stage1.benchmark [--universe-size N] [--k K]
```
比較 **Top-K 基準 / 自創 heuristic / MILP 最佳解**,輸出 optimality gap + 求解時間 → `output/benchmark.json`。`--universe-size N` 把 universe 裁成 μ 前 N 檔(做 N=10/50/100 規模掃描用),`--k` 覆寫持股檔數。

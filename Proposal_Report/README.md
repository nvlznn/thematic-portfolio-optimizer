# Final Report 草稿

> NTU 114-2 作業研究期末專案的「Written Report」草稿，
> 對應作業要求文件 `Source_from_cool/OR114-2_finalProject_requirement.pdf` §5。
> 範本風格參考 `Source_from_cool/OR112-2_proposalTemplateByTAs/`。

## 檔案

| 檔案 | 用途 |
|---|---|
| `final_report.tex` | 主檔（XeLaTeX + xeCJK） |
| `figures/` | 報告引用圖片放這裡（NAV 曲線、權重 heatmap 等回測產出） |

## 編譯

```bash
# 需要 XeLaTeX（macOS：MacTeX；Linux：texlive-xetex；Windows：MikTeX）
xelatex final_report.tex
xelatex final_report.tex   # 第二次為了 cleveref 的 ref 更新
```

如系統沒有 `PingFang TC`，請把 `\setCJKmainfont{PingFang TC}` 改成系統實際有的中文字型，例如 `Heiti TC`、`新細明體`、`Noto Sans CJK TC`。

## 報告結構（對應作業 §5）

1. **Introduction** — 背景、動機、痛點、決策者、權衡、為何是 OR
2. **Problem description** — 兩階段業務語言描述、人工 → 系統化對照表
3. **Mathematical model** — Stage 1 / Stage 2 完整 MILP 公式
4. **Algorithms** — 自創 greedy heuristic（pseudocode）＋ Top-K baseline
5. **Data collection and generation** — TEJ 真實資料管線、self-generated 規模掃描
6. **Performance evaluation** — 五個規模的 MILP / heuristic / Top-K 對照表、Stage 2 order sheet 摘要、RQ 回答
7. **Conclusions** — 結論 + 後續工作

12 頁限制：目前草稿配置約 12 頁，若超出可裁減 §6 部分文字 / 縮短 §2 對照表。

## 還需要填補的內容（草稿中以「待補」或佔位符標示）

- **Group ID**：sign-up 後 TA 會配發，請填到 `\date{...}` 處
- **學號**：四位組員的學號填到 `\date{...}` 處
- **組員姓名**：目前只填了劉威廷、闕以諾兩位，另兩位需補上（資料工程與 Stage 2/回測負責人）
- **圖片**：報告目前是純文字＋表格；若 Module 5 回測完成，建議補：
  - NAV 累積報酬曲線（三條策略對比）
  - 權重 heatmap（rolling 後）
  - Beta / CVaR over time

## 重現報告中的數字

```bash
# 從專案根目錄
.venv/bin/python -m MILP_phase_1.stage1.benchmark                       # 全 universe
.venv/bin/python -m MILP_phase_1.stage1.benchmark --universe-size 50  --k 15
.venv/bin/python -m MILP_phase_1.stage1.benchmark --universe-size 100 --k 15
.venv/bin/python -m MILP_phase_1.stage1.benchmark --universe-size 200 --k 20
.venv/bin/python -m MILP_phase_1.stage1.benchmark --universe-size 500 --k 20
.venv/bin/python -m MILP_phase_1.stage1.solve                            # Stage 1 主求解
.venv/bin/python -m MILP_phase_2.stage2.solve                            # Stage 2 主求解
```

Stage 2 的 order sheet 在 `MILP_phase_2/output/order_sheet.csv`，diagnostics 在 `MILP_phase_2/output/stage2_diagnostics.json`。

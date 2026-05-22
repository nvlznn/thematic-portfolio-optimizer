# 第一階段 MILP — 理想連續權重組合

Stage 1：在報酬、風險(Beta/CVaR)、產業集中度、流動性、周轉率限制下,求理想 **連續權重** `w*` 與持股決策 `z*`。

- **輸入**：`data/processed/{prices,candidates,params}.parquet`(§7 介面契約)
- **輸出**：`output/ideal_portfolio.json`(`w*`, `z*`, diagnostics)→ 餵給 Stage 2

> 模型原理見 [model_explain.md](model_explain.md);介面契約、設計細節與設定見 [detail.md](detail.md)。

## 安裝

```bash
pip install pyarrow pandas numpy pulp pyyaml   # 或 pip install -r ../requirements.txt
```

## 使用

在 **專案根目錄** 執行,預設參數已串好,免參數即可跑:

```bash
# 求解 → output/ideal_portfolio.json
python -m MILP_phase_1.stage1.solve

# heuristic vs 最佳解 benchmark(作業 §5.6)
python -m MILP_phase_1.stage1.benchmark                    # 全 universe
python -m MILP_phase_1.stage1.benchmark --universe-size 50 --k 10
```

覆寫預設路徑用 `--config` / `--processed` / `--output`。

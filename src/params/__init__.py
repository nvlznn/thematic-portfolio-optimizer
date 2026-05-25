"""Module 2 — 參數估計(B 負責)。

把 A 產出的 prices/candidates/params + TEJ 原始檔轉成 Stage 1 MILP 需要的參數:
μ_i(吸引力)、β_i(系統風險)、ℓ_i(流動性上限)、r_{i,s}(CVaR 情境)。

* :mod:`mu`             — 動能版 + 財務版(動能 z-score + ROE z-score)
* :mod:`beta`          — 自算滾動 OLS(vs TAIEX)或沿用 TEJ CAPM Beta
* :mod:`cvar_scenarios`— 歷史情境 + block bootstrap
* :mod:`liquidity`     — ADV20 → ℓ_i
* :mod:`build`         — 組成 factors.parquet(§7.3)+ scenarios.parquet(§7.4)
"""

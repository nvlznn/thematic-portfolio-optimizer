"""Economic sanity 檢查：價格 / β / μ / 流動性 / 相關矩陣是否合理。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_instance(instance_dir: Path) -> dict:
    errs: list[str] = []
    warns: list[str] = []

    prices = pd.read_parquet(instance_dir / "prices.parquet")
    cands = pd.read_parquet(instance_dir / "candidates.parquet")
    params = pd.read_parquet(instance_dir / "params.parquet")
    scen = pd.read_parquet(instance_dir / "scenarios.parquet")
    spec = json.loads((instance_dir / "instance_spec.json").read_text())

    # 1. 價格與量
    if (prices["close"] <= 0).any():
        errs.append("prices.close ≤ 0 存在")
    if prices[["close", "volume", "amount"]].isna().any().any():
        errs.append("prices 關鍵欄位有 NaN")

    # 2. β 範圍
    beta = params.drop_duplicates("stock_id")["beta_3m"]
    if not ((beta >= -1.0) & (beta <= 3.0)).all():
        errs.append(f"beta 越界：min={beta.min()}, max={beta.max()}")

    # 3. μ_override z-score 性質
    if "mu_override" in params.columns:
        mu = params.drop_duplicates("stock_id")["mu_override"]
        if abs(mu.mean()) > 0.5 or not (0.5 < mu.std() < 2.0):
            warns.append(f"mu_override 非 z-scored: mean={mu.mean():.3f}, std={mu.std():.3f}")

    # 4. 市值門檻
    n_pass = int((cands["market_cap"] >= 5e9).sum())
    if n_pass < 0.95 * len(cands):
        warns.append(f"僅 {n_pass}/{len(cands)} 過 min_market_cap")

    # 5. ADV 正性
    adv = prices.groupby("stock_id")["amount"].mean()
    if (adv <= 0).any():
        errs.append("某些股票的平均成交金額 ≤ 0")

    # 6. F5 相關矩陣 sanity（粗略）
    log_ret = scen.values
    if log_ret.shape[1] > 1:
        corr = np.corrcoef(log_ret.T)
        off_diag_mean = float((corr.sum() - np.trace(corr)) / (corr.shape[0] * (corr.shape[0] - 1)))
        # independent: 預期接近 0；block/factor: 應該 > 0
        if spec["corr_structure"] == "independent" and abs(off_diag_mean) > 0.20:
            warns.append(f"independent 但平均相關 = {off_diag_mean:.3f}（預期 ≈ 0）")
        if spec["corr_structure"] in ("block", "factor") and off_diag_mean < 0.05:
            warns.append(f"{spec['corr_structure']} 但平均相關 = {off_diag_mean:.3f}（預期 > 0）")

    # 7. F3 產業分佈
    if spec["industry_shape"] == "5_concentrated":
        ind_counts = cands["industry"].value_counts(normalize=True).sort_values(ascending=False)
        if len(ind_counts) >= 1 and ind_counts.iloc[0] < 0.4:
            warns.append(f"5_concentrated 但最大產業佔比 = {ind_counts.iloc[0]:.2%}（預期 ≥ 50%）")

    return {"errors": errs, "warnings": warns, "ok": len(errs) == 0}

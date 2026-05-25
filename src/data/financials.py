"""Step 0 — 把 TEJ 財務原始檔接進 params.parquet 的 roe / revenue_growth 欄。

`data/raw/<YYYYMM>/raw_financials.csv` 內含每檔每季的「營收成長率」與「ROE(A)稅後」,
但 `data/processed/params.parquet` 的 `roe` / `revenue_growth` 兩欄目前全空。本腳本:

1. 解析季頻財務檔(YYYYMM = 季末月,如 202506 = 2025 Q2)。
2. 套用「公布時滯」(預設 90 天)算出每季資料真正可用的 effective_date,避免前視偏誤。
3. 用 merge_asof(backward)把「截至該日最近一筆已公布財務」貼到 params 每個日頻列。
4. 寫回 params.parquet(只動 roe / revenue_growth,date 欄原樣保留)。

執行(專案根目錄)::

    python src/data/financials.py
    python src/data/financials.py --lag-days 90

冪等:可重複執行,每次都會用最新的 raw_financials.csv 重算覆蓋。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# raw_financials.csv 欄位順序固定(中文表頭在某些編碼下會亂碼,改用「位置」對應):
#   0: 證券代碼+名稱   1: 年季(YYYYMM)   2: 營收成長率   3: ROE(A)稅後
RAW_COLUMNS = ["code_name", "period", "revenue_growth", "roe"]


def _read_tej_csv(path: Path) -> pd.DataFrame:
    """TEJ 匯出檔的編碼/分隔不一致,逐一嘗試直到欄數合理。"""
    for encoding, sep in [("utf-16", "\t"), ("utf-8", "\t"), ("cp950", ","), ("utf-8", ",")]:
        try:
            df = pd.read_csv(path, encoding=encoding, sep=sep, quoting=3)
        except Exception:  # noqa: BLE001
            continue
        if df.shape[1] >= 4:
            return df
    raise RuntimeError(f"無法解析 {path}(編碼/分隔皆失敗)")


def _decode_dates(series: pd.Series) -> pd.Series:
    """params.parquet 的 date 欄可能是「YYYYMMDD 被當成 ns since epoch」,在此還原成真實日期。"""
    as_int = series.astype("int64")
    if (as_int >= 10**7).all() and (as_int < 10**9).all():
        return pd.to_datetime(as_int.astype(str), format="%Y%m%d")
    return pd.to_datetime(series)


def _latest_raw_dir(raw_root: Path) -> Path:
    """挑 data/raw 底下最新的 YYYYMM 子目錄。"""
    subdirs = [p for p in raw_root.iterdir() if p.is_dir() and re.fullmatch(r"\d{6}", p.name)]
    if not subdirs:
        raise FileNotFoundError(f"{raw_root} 下找不到 YYYYMM 子目錄")
    return max(subdirs, key=lambda p: p.name)


def load_financials(path: Path, lag_days: int) -> pd.DataFrame:
    """讀季頻財務檔 → 回傳 (stock_id, effective_date, roe, revenue_growth)。"""
    raw = _read_tej_csv(path)
    raw = raw.iloc[:, :4].copy()
    raw.columns = RAW_COLUMNS

    stock_id = raw["code_name"].astype(str).str.strip().str.split(r"\s+").str[0]
    quarter_end = pd.to_datetime(raw["period"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    effective_date = quarter_end + pd.Timedelta(days=lag_days)

    out = pd.DataFrame(
        {
            "stock_id": stock_id,
            "effective_date": effective_date,
            "roe": pd.to_numeric(raw["roe"], errors="coerce"),
            "revenue_growth": pd.to_numeric(raw["revenue_growth"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["stock_id", "effective_date"])
    out = out[out["roe"].notna() | out["revenue_growth"].notna()]
    return out.sort_values("effective_date").reset_index(drop=True)


def enrich_params(params_path: Path, financials: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """把每檔最近一筆已公布財務(effective_date ≤ 該日)貼到 params 每個日頻列。"""
    params = pd.read_parquet(params_path)
    params = params.reset_index(drop=True)

    # 資料品質修正:params 的 stock_id 被污染成「代碼+名稱」(如 "0050 元大台灣50"),
    # 與 prices/candidates 的純代碼對不上,會讓 beta_3m 等所有 join 失配。先正規化成純代碼。
    params["stock_id"] = params["stock_id"].astype(str).str.strip().str.split(r"\s+").str[0]

    params["_row"] = range(len(params))
    params["_real_date"] = _decode_dates(params["date"])

    left = params.sort_values("_real_date")
    merged = pd.merge_asof(
        left,
        financials,
        left_on="_real_date",
        right_on="effective_date",
        by="stock_id",
        direction="backward",
    )

    merged["roe"] = merged["roe_y"] if "roe_y" in merged else merged["roe"]
    merged["revenue_growth"] = (
        merged["revenue_growth_y"] if "revenue_growth_y" in merged else merged["revenue_growth"]
    )

    # 還原原始列順序,丟掉所有輔助欄
    merged = merged.sort_values("_row")
    drop_cols = [c for c in merged.columns if c.endswith("_x") or c.endswith("_y")]
    drop_cols += ["_row", "_real_date", "effective_date"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    merged = merged[params.drop(columns=["_row", "_real_date"]).columns]

    stats = {
        "roe_non_null": int(merged["roe"].notna().sum()),
        "revenue_growth_non_null": int(merged["revenue_growth"].notna().sum()),
        "rows": len(merged),
    }
    return merged.reset_index(drop=True), stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 raw_financials.csv 接進 params.parquet")
    parser.add_argument("--raw-dir", type=Path, default=None, help="預設自動取 data/raw 最新 YYYYMM")
    parser.add_argument("--params", type=Path, default=Path("data/processed/params.parquet"))
    parser.add_argument("--lag-days", type=int, default=90, help="財報公布時滯(避免前視偏誤)")
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir or _latest_raw_dir(Path("data/raw"))
    fin_path = raw_dir / "raw_financials.csv"
    print(f"[financials] 讀取財務檔:{fin_path}")
    financials = load_financials(fin_path, lag_days=args.lag_days)
    print(
        f"[financials] 季頻財務 {len(financials)} 列、{financials['stock_id'].nunique()} 檔、"
        f"季別 {sorted(pd.to_datetime(financials['effective_date']).dt.to_period('Q').astype(str).unique())}"
    )

    print(f"[financials] 套用公布時滯 {args.lag_days} 天,貼到 {args.params}")
    enriched, stats = enrich_params(args.params, financials)
    enriched.to_parquet(args.params, index=False)
    print(
        f"[financials] 完成:roe 非空 {stats['roe_non_null']} / {stats['rows']}、"
        f"revenue_growth 非空 {stats['revenue_growth_non_null']} / {stats['rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

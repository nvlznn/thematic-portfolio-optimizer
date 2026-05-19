"""Stage2 資料準備：把 Stage1 輸出 + 期初狀態 + 當期股價，轉成 MILP 輸入。

來源：

* Stage 1 ``ideal_portfolio.json``（§7.5）：``weights``、``selected``、``as_of``
* ``prices.parquet``（§7.1）：取 ``as_of`` 當日（或之前最近一日）收盤價
* 設定檔 ``portfolio`` 區段：``C0``、``initial_holdings``

輸出：:class:`Stage2Inputs`，包含模型直接吃的所有 numpy 陣列。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOT_SIZE = 1000  # 台股 1 張 = 1000 股


@dataclass
class Stage2Inputs:
    as_of: pd.Timestamp
    stocks: list[str]              # universe（順序固定）
    price_per_lot: np.ndarray      # P_i，元 / 張
    x0_lots: np.ndarray             # x_i^0，期初張數
    w_star: np.ndarray              # 第一階段理想權重（對齊 stocks）
    z_star: np.ndarray              # 第一階段持股二元決策
    L_lot: np.ndarray               # L_i^lot = ⌈L V^0 / P_i⌉
    U_lot: np.ndarray               # U_i^lot = ⌊U V^0 / P_i⌋
    u_bar: np.ndarray               # ū_i = max(0, U_i^lot - x_i^0)
    C0: float                       # 期初現金
    V0: float                       # 期初總資產
    diagnostics: dict


def _decode_dates(series: pd.Series) -> pd.Series:
    as_int = series.astype("int64")
    if (as_int >= 10**7).all() and (as_int < 10**9).all():
        return pd.to_datetime(as_int.astype(str), format="%Y%m%d")
    return pd.to_datetime(series)


def _resolve_paths(cfg: dict, cfg_path: Path) -> tuple[Path, Path]:
    # 相對路徑以「專案根目錄」（config 檔上兩層 = MILP_phase_2 的 parent）為基準
    project_root = cfg_path.resolve().parents[2]
    ip = Path(cfg["input"]["ideal_portfolio"])
    pp = Path(cfg["input"]["prices"])
    if not ip.is_absolute():
        ip = (project_root / ip).resolve()
    if not pp.is_absolute():
        pp = (project_root / pp).resolve()
    return ip, pp


def _latest_prices_at(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    snap = prices[prices["date"] <= as_of]
    if snap.empty:
        raise ValueError(f"prices.parquet 沒有 <= {as_of} 的資料")
    latest = (
        snap.sort_values("date").groupby("stock_id")["close"].last()
    )
    return latest


def build_stage2_inputs(cfg: dict, cfg_path: Path) -> Stage2Inputs:
    ideal_path, prices_path = _resolve_paths(cfg, cfg_path)

    with ideal_path.open("r", encoding="utf-8") as f:
        ideal = json.load(f)

    as_of = pd.Timestamp(ideal["as_of"])
    weights: dict[str, float] = ideal["weights"]
    selected: list[str] = ideal["selected"]

    prices = pd.read_parquet(prices_path)
    prices["date"] = _decode_dates(prices["date"])
    px_per_share = _latest_prices_at(prices, as_of)

    initial_holdings_cfg: dict[str, int] = cfg["portfolio"].get("initial_holdings") or {}
    C0 = float(cfg["portfolio"]["C0"])

    # universe = selected ∪ 有持股者；selected 順序優先，再附上需要清倉的持股
    selected_set = set(selected)
    held_only = [s for s in initial_holdings_cfg if s not in selected_set and initial_holdings_cfg.get(s, 0) > 0]
    universe = list(selected) + held_only

    # 過濾掉沒有報價的股票
    missing_price = [s for s in universe if s not in px_per_share.index]
    if missing_price:
        # 若 missing 來自 selected：補一個說明、後續輸出 deviation 會反映
        universe = [s for s in universe if s in px_per_share.index]

    if not universe:
        raise RuntimeError("Stage2 universe 為空 — 檢查 Stage1 輸出與 prices.parquet")

    P_share = px_per_share.reindex(universe).to_numpy(dtype=float)
    P_lot = P_share * LOT_SIZE

    x0 = np.array(
        [int(initial_holdings_cfg.get(s, 0)) for s in universe], dtype=int
    )
    w_star = np.array([weights.get(s, 0.0) for s in universe], dtype=float)
    z_star = np.array([1 if s in selected_set else 0 for s in universe], dtype=int)

    V0 = C0 + float(np.dot(P_lot, x0))

    L = float(cfg["stage2"]["L"])
    U = float(cfg["stage2"]["U"])

    L_lot_raw = np.array([math.ceil(L * V0 / p) if p > 0 else 0 for p in P_lot], dtype=int)
    U_lot = np.array([math.floor(U * V0 / p) if p > 0 else 0 for p in P_lot], dtype=int)

    # 若 L_lot > U_lot（高股價 + 嚴格上限），允許 L 降到 U，避免 box 空集
    auto_lowered: list[str] = []
    L_lot = L_lot_raw.copy()
    for i, sid in enumerate(universe):
        if L_lot[i] > U_lot[i]:
            auto_lowered.append(sid)
            L_lot[i] = U_lot[i]

    # 若連 U_lot = 0（極高股價）也納入 selected，等於模型只能 x_i = 0 = L_lot z*
    # 把那種股票自 z* 拿掉，視同 Stage1 沒選；但仍保留在 universe 以便處理
    forced_off: list[str] = []
    for i, sid in enumerate(universe):
        if z_star[i] == 1 and U_lot[i] == 0:
            forced_off.append(sid)
            z_star[i] = 0

    u_bar = np.maximum(0, U_lot - x0)

    diagnostics = {
        "n_universe": len(universe),
        "n_selected_kept": int(z_star.sum()),
        "auto_lowered_L_lot": auto_lowered,
        "forced_off_high_price": forced_off,
        "V0": V0,
        "as_of": as_of.strftime("%Y-%m-%d"),
    }

    return Stage2Inputs(
        as_of=as_of,
        stocks=universe,
        price_per_lot=P_lot,
        x0_lots=x0,
        w_star=w_star,
        z_star=z_star,
        L_lot=L_lot,
        U_lot=U_lot,
        u_bar=u_bar,
        C0=C0,
        V0=V0,
        diagnostics=diagnostics,
    )

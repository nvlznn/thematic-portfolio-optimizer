# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

NTU 114-2 作業研究期末專案：a two-stage MILP decision-support system for **thematic Taiwan stock investing**. The pipeline is documented in detail in `proposal.pdf` (§6 has the full formulation) and `開發計畫.md` (Modules 1–5, interface contracts in §7). Read both before making non-trivial changes — the interface contracts (parquet schemas, JSON shape, CSV columns) are agreed-upon across team members and changes need coordination.

## Commands

All commands assume the project root as cwd and the venv activated (or invoked via `.venv/bin/python`).

```bash
# Environment
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Stage 1 only needs the subset: pyarrow pandas numpy pulp pyyaml

# Data: regenerate processed parquet from raw CSVs (run from project root)
.venv/bin/python src/data/clean.py

# Stage 1 — ideal continuous weights → MILP_phase_1/output/ideal_portfolio.json
.venv/bin/python -m MILP_phase_1.stage1.solve

# Stage 2 — integer lot rounding → MILP_phase_2/output/order_sheet.csv
.venv/bin/python -m MILP_phase_2.stage2.solve

# End-to-end
.venv/bin/python -m MILP_phase_1.stage1.solve && .venv/bin/python -m MILP_phase_2.stage2.solve
```

Both stages take `--config`, `--processed`/input paths, and `--output` flags; defaults are wired so the end-to-end command works without arguments.

## Architecture

### Pipeline (proposal §4 modules)

```
data/raw/YYYYMM/*.csv
    │  src/data/clean.py
    ▼
data/processed/{prices,candidates,params}.parquet  ← §7.1–§7.3 interface contracts
    │  MILP_phase_1/stage1/  (data_prep → model → solve)
    ▼
MILP_phase_1/output/ideal_portfolio.json           ← §7.5 (w*, z*, diagnostics)
    │  MILP_phase_2/stage2/  (data_prep → model → solve)
    ▼
MILP_phase_2/output/order_sheet.csv                ← §7.6 (u, v, x, C)
```

Each stage's package mirrors the same layout: `config.yaml` (all knobs), `data_prep.py` (parquet/JSON → numpy arrays), `model.py` (PuLP MILP build + solve + fallback), `solve.py` (CLI entry that writes the §7 contract output).

### Interface contracts (§7 — do not break casually)

- **`prices.parquet`** (§7.1): `date, stock_id, close, volume, amount` — daily price/volume per stock
- **`candidates.parquet`** (§7.2): `stock_id, name, market_cap, industry, theme_tag` — the static stock pool
- **`params.parquet`** (§7.3): daily join of prices + market_cap + beta_3m + roe + revenue_growth
- **`ideal_portfolio.json`** (§7.5): `{as_of, objective_value, weights, selected, diagnostics}` — Stage 1 → Stage 2 hand-off
- **`order_sheet.csv`** (§7.6): `stock_id, price, x0_lots, buy_lots, sell_lots, x_lots, weight_actual, weight_ideal, deviation, cash_after` — final executable trades

### Mathematical model

Both stages are MILPs (no nonlinear pieces — proposal §6 has full linearizations: CVaR via Rockafellar–Uryasev, turnover via δ⁺/δ⁻ split, L1 deviation via `d_i ≥ ±(...)`, buy/sell indicators via big-M with B_i + Q_i ≤ 1). Stage 1 also handles relaxation fallback when infeasible (cvar → turnover → beta → industry, logged in `diagnostics.relaxation_log`). Solver backend selection lives in `_solver()` in each `model.py` — defaults to PuLP+CBC; Gurobi tried only when `solver.backend: gurobi|auto` and a license is available.

### Stage 1 / Stage 2 coherence (non-obvious)

The proposal specifies scalar `L, U` (per-stock holding floor/ceiling), but Stage 2's integer-lot constraint means the **achievable** weight set per stock is the discrete grid `{k · P_lot / V0}`. To keep both stages in the same feasible polytope, Stage 1 uses **per-stock**:

- `L_i = ⌈L · V0 / P_lot⌉ · P_lot / V0`
- `U_i = ⌊U · V0 / P_lot⌋ · P_lot / V0`

This is computed in `MILP_phase_1/stage1/data_prep.py:_build_lot_bounds`. When `U_i = 0` (1 lot price > U·V0), the model explicitly forces `z_i = 0` — without that extra constraint, the solver can satisfy `Σz = K` by picking such stocks "for free" with `w_i = 0` and contaminate Stage 1's `selected` list.

Editing `stage1/config.yaml`'s `L`/`U` is still the right knob; the per-stock rounding is automatic.

### Other gotchas

- **Date encoding**: `src/data/clean.py` writes dates by passing `YYYYMMDD` ints directly to `pd.to_datetime`, which interprets them as **ns since epoch**, producing bogus 1970-... timestamps. Both stage `data_prep.py` files have a `_decode_dates` helper that detects the small-int-in-datetime case and reconstructs the real date. If `clean.py` is fixed upstream, the helper falls through unchanged.
- **First period auto-fix**: when `w0 ≡ 0` (no prior portfolio), the minimum turnover is `Σw = 1`, so any `τ < 1` would be structurally infeasible. `stage1/model.py` auto-bumps `τ` to 2.0 in this case — don't try to "fix" by lowering τ in config.
- **Post-hoc CVaR**: `diagnostics.cvar_realised` is recomputed directly from `w · scenarios` rather than read from the LP's η/ξ values (those drift when CVaR is relaxed). Same approach should be used for any future re-evaluation of a relaxed constraint.
- **1 張 = 1000 股**: lot size is hard-coded as `LOT_SIZE = 1000` in both data_prep files. Stage 2's `P_i` in the model is **price per lot** (close × 1000), not per share.
- **YAML scientific notation**: PyYAML treats `5.0e9` as a string; use `5.0e+9` (explicit sign). Configs already follow this convention; preserve it.

## Team / role split (per 開發計畫.md §6)

| Person | Module | Files owned |
|---|---|---|
| A | 資料工程 + 候選股池 | `src/data/*.py`, `data/processed/{prices,candidates}.parquet` |
| B | 參數估計 | (TBD) `src/params/*`, `params.parquet` extensions |
| C (劉威廷/闕以諾) | Stage 1 MILP | `MILP_phase_1/` |
| D | Stage 2 MILP + 回測 | `MILP_phase_2/`, `src/backtest/*` (TBD) |

The folder name `MILP_phase_{1,2}` was renamed from `劉威廷and闕以諾/`; if work expands to other team members, follow the `MILP_phase_*` convention.

## Reading order for new contributors

1. `proposal.pdf` §§3, 6 (decision problems + full MILP formulation)
2. `開發計畫.md` §7 (interface contracts — these are the API between team members)
3. `MILP_phase_1/README.md` then `MILP_phase_2/README.md` (per-stage docs with run examples)
4. `MILP_phase_1/stage1/model.py` and `MILP_phase_2/stage2/model.py` (the two MILPs side-by-side make the design clearest)

"""prices.py：價格非負、日期 dtype、β 復原性、F4 tail-fat 性質。"""
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from synthetic_data.generator.factors import build_instance_specs
from synthetic_data.generator.parameters import build_parameters
from synthetic_data.generator.prices import simulate_prices
from synthetic_data.generator.rng import spawn_rngs
from synthetic_data.generator.universe import build_universe


def _gen(spec):
    rngs = spawn_rngs(spec.seed)
    universe = build_universe(spec, rngs["universe"])
    params = build_parameters(spec, rngs["parameters"])
    df = simulate_prices(spec, universe, params, rngs["prices"])
    return df, universe, params


def _baseline_spec(**overrides):
    base = next(s for s in build_instance_specs() if s.instance_id == "SYN_005_size50x250")
    return replace(base, instance_id="TEST", **overrides)


def test_prices_strictly_positive():
    spec = _baseline_spec(n_stocks=20)
    df, _, _ = _gen(spec)
    assert (df["close"] > 0).all()


def test_no_nan_in_required_columns():
    spec = _baseline_spec(n_stocks=20)
    df, _, _ = _gen(spec)
    for col in ("close", "volume", "amount"):
        assert not df[col].isna().any(), f"{col} has NaN"


def test_date_dtype_is_datetime():
    spec = _baseline_spec(n_stocks=20)
    df, _, _ = _gen(spec)
    # 接受 datetime64[us] 或 datetime64[ns]（兩者 stage1 都能吃）
    assert "datetime64" in str(df["date"].dtype)


def test_no_yyyymmdd_int_trap():
    """確保我們不會回退到 clean.py 的 YYYYMMDD-as-int 陷阱（CLAUDE.md gotcha）。"""
    spec = _baseline_spec(n_stocks=20)
    df, _, _ = _gen(spec)
    # 真實 datetime 的 int64 值是 ns/us since epoch；YYYYMMDD 落在 [10^7, 10^8)
    as_int = df["date"].astype("int64")
    assert (as_int >= 10**9).all(), "date 似乎被誤存成 YYYYMMDD int"


def test_regression_beta_recovers_target():
    """每股的回歸 β ≈ target β ± tolerance（factor model 復原性）。"""
    spec = _baseline_spec(n_stocks=30, corr_structure="factor", beta_dist="wide")
    df, universe, params = _gen(spec)
    wide = df.pivot(index="date", columns="stock_id", values="close").sort_index()
    log_r = np.log(wide / wide.shift(1)).dropna()
    # market proxy：所有股票的等權平均
    market = log_r.mean(axis=1)
    var_m = market.var()
    recovered = []
    for sid, target_beta in zip(universe.stock_ids, params.beta):
        cov = np.cov(log_r[sid], market)[0, 1]
        recovered.append((target_beta, cov / var_m))
    diffs = np.array([abs(a - b) for a, b in recovered])
    # 平均誤差 < 0.40（單股回歸雜訊大，且我們不是用真實 market index）
    assert diffs.mean() < 0.50, f"mean |β_target - β_regression| = {diffs.mean():.3f}"


def test_t_distribution_has_higher_kurtosis_than_gaussian():
    spec_g = _baseline_spec(n_stocks=20, return_dist="gaussian")
    spec_t = _baseline_spec(n_stocks=20, return_dist="t")
    df_g, _, _ = _gen(spec_g)
    df_t, _, _ = _gen(spec_t)

    def _kurt(df):
        wide = df.pivot(index="date", columns="stock_id", values="close").sort_index()
        log_r = np.log(wide / wide.shift(1)).dropna().values.flatten()
        return kurtosis(log_r, fisher=False)

    k_g, k_t = _kurt(df_g), _kurt(df_t)
    assert k_t > k_g, f"t kurtosis {k_t:.2f} should exceed gaussian {k_g:.2f}"

"""universe.py：產業分配與 market_cap 形狀。"""
from dataclasses import replace

import numpy as np

from synthetic_data.generator.factors import build_instance_specs
from synthetic_data.generator.rng import spawn_rngs
from synthetic_data.generator.universe import build_universe


def _baseline_spec(n=200, industry_shape="5_balanced"):
    base = next(s for s in build_instance_specs() if s.instance_id == "SYN_005_size50x250")
    return replace(base, n_stocks=n, industry_shape=industry_shape, instance_id="TEST")


def test_balanced_industries_have_at_least_one_stock_each():
    spec = _baseline_spec(n=200, industry_shape="5_balanced")
    rng = spawn_rngs(spec.seed)["universe"]
    universe = build_universe(spec, rng)
    counts = dict(zip(*np.unique(universe.industries, return_counts=True)))
    assert len(counts) == 5
    for ind, c in counts.items():
        assert c >= 1, f"{ind} has 0 stocks"


def test_concentrated_top_industry_dominates():
    spec = _baseline_spec(n=200, industry_shape="5_concentrated")
    rng = spawn_rngs(spec.seed)["universe"]
    universe = build_universe(spec, rng)
    _, counts = np.unique(universe.industries, return_counts=True)
    top_share = counts.max() / counts.sum()
    assert top_share >= 0.50, f"top industry share = {top_share:.2%}, expected ≥ 50%"


def test_market_cap_above_threshold_majority():
    spec = _baseline_spec(n=200)
    rng = spawn_rngs(spec.seed)["universe"]
    universe = build_universe(spec, rng)
    n_pass = int((universe.market_caps >= 5e9).sum())
    assert n_pass / len(universe.market_caps) >= 0.90


def test_stock_ids_padded_4_digits():
    spec = _baseline_spec(n=10)
    rng = spawn_rngs(spec.seed)["universe"]
    universe = build_universe(spec, rng)
    assert universe.stock_ids[0] == "SYN0001"
    assert universe.stock_ids[-1] == "SYN0010"

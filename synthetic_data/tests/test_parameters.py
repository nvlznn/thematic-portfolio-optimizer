"""parameters.py：β / μ 的統計性質（含 F6/F7 control）。"""
from dataclasses import replace

import numpy as np

from synthetic_data.generator.factors import build_instance_specs
from synthetic_data.generator.parameters import build_parameters
from synthetic_data.generator.rng import spawn_rngs


def _spec_for(beta_dist="tight", mu_beta_corr=0.0, n=300):
    base = next(s for s in build_instance_specs() if s.instance_id == "SYN_005_size50x250")
    return replace(base, n_stocks=n, beta_dist=beta_dist, mu_beta_corr=mu_beta_corr, instance_id="TEST")


def test_beta_in_valid_range():
    spec = _spec_for("wide", n=500)
    rng = spawn_rngs(spec.seed)["parameters"]
    params = build_parameters(spec, rng)
    assert params.beta.min() >= -1.0
    assert params.beta.max() <= 3.0


def test_beta_tight_concentrates_around_one():
    spec = _spec_for("tight", n=500)
    rng = spawn_rngs(spec.seed)["parameters"]
    params = build_parameters(spec, rng)
    assert 0.8 <= params.beta.mean() <= 1.2
    assert params.beta.std() < 0.30


def test_beta_wide_has_higher_spread():
    spec_t = _spec_for("tight", n=500)
    spec_w = _spec_for("wide", n=500)
    pt = build_parameters(spec_t, spawn_rngs(spec_t.seed)["parameters"])
    pw = build_parameters(spec_w, spawn_rngs(spec_w.seed)["parameters"])
    assert pw.beta.std() > pt.beta.std()


def test_mu_is_z_scored():
    spec = _spec_for(n=500)
    rng = spawn_rngs(spec.seed)["parameters"]
    params = build_parameters(spec, rng)
    assert abs(params.mu_override.mean()) < 0.2
    assert 0.7 < params.mu_override.std() < 1.3


def test_mu_beta_positive_correlation_realised():
    spec = _spec_for(mu_beta_corr=0.5, n=500)
    rng = spawn_rngs(spec.seed)["parameters"]
    params = build_parameters(spec, rng)
    corr = float(np.corrcoef(params.mu_override, params.beta)[0, 1])
    assert 0.30 <= corr <= 0.70, f"corr={corr:.3f}, expected ≈ +0.5"


def test_mu_beta_negative_correlation_realised():
    spec = _spec_for(mu_beta_corr=-0.5, n=500)
    rng = spawn_rngs(spec.seed)["parameters"]
    params = build_parameters(spec, rng)
    corr = float(np.corrcoef(params.mu_override, params.beta)[0, 1])
    assert -0.70 <= corr <= -0.30, f"corr={corr:.3f}, expected ≈ -0.5"


def test_mu_beta_zero_correlation_realised():
    spec = _spec_for(mu_beta_corr=0.0, n=500)
    rng = spawn_rngs(spec.seed)["parameters"]
    params = build_parameters(spec, rng)
    corr = float(np.corrcoef(params.mu_override, params.beta)[0, 1])
    assert abs(corr) < 0.20, f"corr={corr:.3f}, expected ≈ 0"


def test_adv_positive():
    spec = _spec_for(n=500)
    rng = spawn_rngs(spec.seed)["parameters"]
    params = build_parameters(spec, rng)
    assert (params.adv20_amount > 0).all()

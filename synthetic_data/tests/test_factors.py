"""factors.py：DoE 矩陣完整性與一致性。"""
from synthetic_data.generator.factors import build_instance_specs


def test_specs_count_is_27():
    assert len(build_instance_specs()) == 27


def test_instance_ids_unique():
    ids = [s.instance_id for s in build_instance_specs()]
    assert len(set(ids)) == len(ids)


def test_seeds_unique():
    seeds = [s.seed for s in build_instance_specs()]
    assert len(set(seeds)) == len(seeds)


def test_scenario_sets_cover_all_five():
    sets = {s.scenario_set for s in build_instance_specs()}
    assert sets == {"S1", "S2", "S3", "S4", "S5"}


def test_K_times_U_has_slack():
    # 對所有實例，K · U ≥ 1.2（20% lot-rounding margin）
    for s in build_instance_specs():
        assert s.K * s.U >= 1.2, f"{s.instance_id}: K·U={s.K * s.U} < 1.2"


def test_determinism_across_invocations():
    a = build_instance_specs()
    b = build_instance_specs()
    assert [(s.instance_id, s.seed) for s in a] == [(s.instance_id, s.seed) for s in b]

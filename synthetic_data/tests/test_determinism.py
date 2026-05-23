"""相同 seed 兩次生成 → 所有 parquet bytes 完全相同。"""
import shutil
from pathlib import Path

from synthetic_data.runners.generate_all import generate_one
from synthetic_data.generator.factors import build_instance_specs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TMP_A = PROJECT_ROOT / "synthetic_data" / "instances" / "_DETERMINISM_A"
TMP_B = PROJECT_ROOT / "synthetic_data" / "instances" / "_DETERMINISM_B"


def _gen(out_root: Path):
    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True)
    spec = next(s for s in build_instance_specs() if s.instance_id == "SYN_001_size20x100")
    return generate_one(spec, out_root)


def test_same_seed_same_parquet_bytes():
    dir_a = _gen(TMP_A)
    dir_b = _gen(TMP_B)
    try:
        for fname in ("prices.parquet", "candidates.parquet", "params.parquet", "scenarios.parquet"):
            a = (dir_a / fname).read_bytes()
            b = (dir_b / fname).read_bytes()
            assert a == b, f"{fname} bytes differ"
    finally:
        shutil.rmtree(TMP_A, ignore_errors=True)
        shutil.rmtree(TMP_B, ignore_errors=True)

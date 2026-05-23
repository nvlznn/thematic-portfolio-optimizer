"""每模組獨立 RNG。

由單一 ``instance_seed`` 經 ``numpy.random.SeedSequence.spawn`` 散播給各生成模組。
重構任一模組的 draw 順序不會影響其他模組，保留 bit-for-bit reproducibility。
"""
from __future__ import annotations

import numpy as np

# 順序固定：新增模組請追加到尾端，不要中間插入，否則所有既有 seed 的下游 draws 會位移
MODULES: tuple[str, ...] = (
    "universe",
    "parameters",
    "prices",
    "scenarios",
    "initial",
    "writers",
)


def spawn_rngs(seed: int) -> dict[str, np.random.Generator]:
    """從 ``seed`` 生 ``len(MODULES)`` 個獨立 RNG，回傳 {模組名: Generator}。"""
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(MODULES))
    return {name: np.random.default_rng(child) for name, child in zip(MODULES, children)}

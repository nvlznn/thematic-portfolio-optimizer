"""合成實例生成器模組。

每個模組吃自己獨立的 ``numpy.random.Generator``，避免重構一個模組打亂其他模組的
draws（見 ``rng.spawn_rngs``）。
"""

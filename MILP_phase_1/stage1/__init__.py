"""Stage1：理想連續權重組合 MILP。

把 ``data/processed/`` 下的介面契約檔（prices/candidates/params）轉成 MILP 輸入，
解出 ``ideal_portfolio.json``（§7.5）。
"""
from .data_prep import Stage1Inputs, build_stage1_inputs
from .model import Stage1Solution, solve

__all__ = ["Stage1Inputs", "Stage1Solution", "build_stage1_inputs", "solve"]

"""Stage2：交易落地與整數張數轉換 MILP。

讀 Stage 1 的 ``ideal_portfolio.json`` + 期初狀態，輸出 §7.6 ``order_sheet.csv``。
"""
from .data_prep import Stage2Inputs, build_stage2_inputs
from .model import Stage2Solution, solve

__all__ = ["Stage2Inputs", "Stage2Solution", "build_stage2_inputs", "solve"]

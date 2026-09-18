"""
Computes Credit Valuation Adjustment (CVA) from an expected exposure
profile, using a constant hazard rate default model.

    Survival(t)      = exp(-h * t)
    Marginal PD(t)   = Survival(t_i-1) - Survival(t_i)
    CVA              = LGD * sum_i [ EE(t_i) * Marginal PD(t_i) * DF(t_i) ]

This is the standard simplified CVA formula used for teaching and for a
first pass estimate. A production CVA engine would use a full credit
curve bootstrapped from CDS spreads rather than a single flat hazard
rate, and would net exposure across the whole counterparty portfolio,
not one trade.
"""

import numpy as np


def hazard_survival_curve(hazard_rate: float, time_grid: np.ndarray) -> np.ndarray:
    return np.exp(-hazard_rate * time_grid)


def compute_cva(
    ee: np.ndarray,
    time_grid: np.ndarray,
    discount_factors: np.ndarray,
    hazard_rate: float,
    lgd: float,
) -> float:
    survival = hazard_survival_curve(hazard_rate, time_grid)
    marginal_pd = -np.diff(survival)  # positive, one entry per interval

    # Exposure and discount factor at the end of each interval
    ee_interval = ee[1:]
    df_interval = discount_factors[1:]

    cva = lgd * np.sum(ee_interval * marginal_pd * df_interval)
    return float(cva)

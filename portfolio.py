"""
A deliberately simple portfolio of one payer interest rate swap, used as
the trade whose mark to market drives counterparty exposure in this demo.

Simplification, stated plainly: swap mark to market is approximated as

    MTM_t = notional * (fixed_rate - r_t) * remaining_duration_t

This is a first order, duration style proxy for how a payer swap's value
moves with rates. It captures the right sign and the right shape (MTM
falls as rates fall, and sensitivity shrinks as the swap ages towards
maturity) but it is not a full swap curve valuation. A real desk would
value each leg against a bootstrapped discount curve. See the README for
the full list of known limitations.
"""

import numpy as np


def swap_mtm_paths(
    rate_paths: np.ndarray,
    notional: float,
    fixed_rate: float,
    dt: float,
) -> np.ndarray:
    """
    rate_paths: (n_paths, n_steps + 1) simulated short rate paths.
    Returns MTM paths of the same shape, from the payer's perspective
    (receives fixed, pays floating, so MTM rises as rates fall).
    """
    n_paths, n_points = rate_paths.shape
    n_steps = n_points - 1
    total_life = n_steps * dt

    time_grid = np.arange(n_points) * dt
    remaining_duration = np.clip(total_life - time_grid, 0.0, None)

    mtm = notional * (fixed_rate - rate_paths) * remaining_duration
    return mtm

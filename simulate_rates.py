"""
Simulates future short rate paths using the Vasicek model:

    dr_t = a * (b - r_t) * dt + sigma * dW_t

This is a standard, well understood mean reverting short rate model.
It is used here purely as the risk driver for a simple swap exposure
example, not as a claim of a production grade term structure model.
"""

import numpy as np


def simulate_vasicek_paths(
    r0: float,
    a: float,
    b: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns an array of shape (n_paths, n_steps + 1) of simulated short
    rate paths, all starting at r0.
    """
    rng = np.random.default_rng(seed)
    rates = np.empty((n_paths, n_steps + 1))
    rates[:, 0] = r0

    shocks = rng.standard_normal((n_paths, n_steps))
    sqrt_dt = np.sqrt(dt)

    for t in range(n_steps):
        drift = a * (b - rates[:, t]) * dt
        diffusion = sigma * sqrt_dt * shocks[:, t]
        rates[:, t + 1] = rates[:, t] + drift + diffusion

    return rates


def discount_factors(rates_path_mean: np.ndarray, dt: float) -> np.ndarray:
    """
    Simple discount factors built from the average simulated short rate
    path, using discrete compounding. Used for CVA discounting.
    """
    cumulative = np.cumsum(rates_path_mean) * dt
    return np.exp(-cumulative)

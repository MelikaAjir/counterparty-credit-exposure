"""
A simple stress test: re-runs the exposure and CVA calculation under a
shocked starting rate and a shocked volatility, and reports the change.
This mirrors, in miniature, the kind of "what happens to our exposure
under a rates shock" question a stress testing or counterparty risk
team asks routinely, though a real stress framework would use a full
set of regulatory or historical scenarios, not one shock.
"""

from dataclasses import dataclass

import numpy as np

from cva import compute_cva
from exposure import compute_exposure_profile
from portfolio import swap_mtm_paths
from simulate_rates import discount_factors, simulate_vasicek_paths


@dataclass
class StressResult:
    base_cva: float
    stressed_cva: float
    pct_change: float


def run_rate_shock_stress(
    base_params: dict,
    rate_shock_bps: float,
    vol_multiplier: float,
    lgd: float,
    hazard_rate: float,
) -> StressResult:
    time_grid = np.arange(base_params["n_steps"] + 1) * base_params["dt"]

    # Base case
    base_rates = simulate_vasicek_paths(
        r0=base_params["r0"],
        a=base_params["a"],
        b=base_params["b"],
        sigma=base_params["sigma"],
        n_paths=base_params["n_paths"],
        n_steps=base_params["n_steps"],
        dt=base_params["dt"],
        seed=base_params["seed"],
    )
    base_mtm = swap_mtm_paths(
        base_rates, base_params["notional"], base_params["fixed_rate"], base_params["dt"]
    )
    base_ee = compute_exposure_profile(base_mtm)["ee"]
    base_df = discount_factors(base_rates.mean(axis=0), base_params["dt"])
    base_cva = compute_cva(base_ee, time_grid, base_df, hazard_rate, lgd)

    # Stressed case: shock the starting rate and scale volatility
    stressed_rates = simulate_vasicek_paths(
        r0=base_params["r0"] + rate_shock_bps / 10_000,
        a=base_params["a"],
        b=base_params["b"],
        sigma=base_params["sigma"] * vol_multiplier,
        n_paths=base_params["n_paths"],
        n_steps=base_params["n_steps"],
        dt=base_params["dt"],
        seed=base_params["seed"],
    )
    stressed_mtm = swap_mtm_paths(
        stressed_rates, base_params["notional"], base_params["fixed_rate"], base_params["dt"]
    )
    stressed_ee = compute_exposure_profile(stressed_mtm)["ee"]
    stressed_df = discount_factors(stressed_rates.mean(axis=0), base_params["dt"])
    stressed_cva = compute_cva(stressed_ee, time_grid, stressed_df, hazard_rate, lgd)

    pct_change = (stressed_cva - base_cva) / base_cva * 100 if base_cva != 0 else float("nan")

    return StressResult(base_cva=base_cva, stressed_cva=stressed_cva, pct_change=pct_change)

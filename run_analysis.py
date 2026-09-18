"""
Runs the full counterparty exposure and CVA pipeline for a single payer
interest rate swap:

    1. Simulate short rate paths (Vasicek).
    2. Value the swap along each path.
    3. Compute Expected Exposure and 95% Potential Future Exposure,
       with and without a simple collateral agreement.
    4. Compute CVA from the expected exposure profile.
    5. Run a rate shock stress test and report the change in CVA.
    6. Save a chart of the exposure profiles.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cva import compute_cva
from exposure import apply_collateral, compute_exposure_profile
from portfolio import swap_mtm_paths
from simulate_rates import discount_factors, simulate_vasicek_paths
from stress import run_rate_shock_stress

# --- Parameters -------------------------------------------------------
PARAMS = dict(
    r0=0.04,        # starting short rate, 4%
    a=0.3,           # mean reversion speed
    b=0.04,          # long run mean rate
    sigma=0.015,     # rate volatility
    n_paths=5000,
    n_steps=20,      # quarterly steps
    dt=0.25,         # quarter year
    seed=7,
)
NOTIONAL = 10_000_000
FIXED_RATE = 0.04
COLLATERAL_THRESHOLD = 250_000
LGD = 0.60
HAZARD_RATE = 0.02  # roughly a BBB rated counterparty


def main():
    time_grid = np.arange(PARAMS["n_steps"] + 1) * PARAMS["dt"]

    rate_paths = simulate_vasicek_paths(
        r0=PARAMS["r0"], a=PARAMS["a"], b=PARAMS["b"], sigma=PARAMS["sigma"],
        n_paths=PARAMS["n_paths"], n_steps=PARAMS["n_steps"], dt=PARAMS["dt"],
        seed=PARAMS["seed"],
    )

    mtm_paths = swap_mtm_paths(rate_paths, NOTIONAL, FIXED_RATE, PARAMS["dt"])

    exposure_result = compute_exposure_profile(mtm_paths)
    collateral_result = apply_collateral(exposure_result["exposure_paths"], COLLATERAL_THRESHOLD)

    df = discount_factors(rate_paths.mean(axis=0), PARAMS["dt"])
    cva_uncollateralised = compute_cva(exposure_result["ee"], time_grid, df, HAZARD_RATE, LGD)
    cva_collateralised = compute_cva(collateral_result["ee_net"], time_grid, df, HAZARD_RATE, LGD)

    print("Counterparty Credit Exposure & CVA, single payer swap")
    print("=" * 56)
    print(f"Peak Expected Exposure (uncollateralised): £{exposure_result['ee'].max():,.0f}")
    print(f"Peak 95% PFE (uncollateralised):            £{exposure_result['pfe_95'].max():,.0f}")
    print(f"Peak Expected Exposure (collateralised):    £{collateral_result['ee_net'].max():,.0f}")
    print(f"Peak 95% PFE (collateralised):               £{collateral_result['pfe_95_net'].max():,.0f}")
    print()
    print(f"CVA, uncollateralised: £{cva_uncollateralised:,.0f}")
    print(f"CVA, collateralised:   £{cva_collateralised:,.0f}")
    print(f"Collateral reduces CVA by: {(1 - cva_collateralised / cva_uncollateralised) * 100:.1f}%")

    stress_result = run_rate_shock_stress(
        base_params={**PARAMS, "notional": NOTIONAL, "fixed_rate": FIXED_RATE},
        rate_shock_bps=-150,  # a 150bp downward rate shock
        vol_multiplier=1.5,
        lgd=LGD,
        hazard_rate=HAZARD_RATE,
    )
    print()
    print("Stress test: -150bp rate shock, 1.5x volatility")
    print(f"Base CVA:      £{stress_result.base_cva:,.0f}")
    print(f"Stressed CVA:  £{stress_result.stressed_cva:,.0f}")
    print(f"Change:        {stress_result.pct_change:+.1f}%")

    # --- Chart ---------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(time_grid, exposure_result["ee"], label="EE, uncollateralised", color="#1F3864")
    plt.plot(time_grid, exposure_result["pfe_95"], label="95% PFE, uncollateralised",
             color="#1F3864", linestyle="--")
    plt.plot(time_grid, collateral_result["ee_net"], label="EE, collateralised", color="#C00000")
    plt.plot(time_grid, collateral_result["pfe_95_net"], label="95% PFE, collateralised",
             color="#C00000", linestyle="--")
    plt.xlabel("Time (years)")
    plt.ylabel("Exposure (£)")
    plt.title("Counterparty Exposure Profile: Single Payer Interest Rate Swap")
    plt.legend()
    plt.tight_layout()
    plt.savefig("charts/exposure_profile.png", dpi=150)
    print()
    print("Chart saved to charts/exposure_profile.png")


if __name__ == "__main__":
    main()

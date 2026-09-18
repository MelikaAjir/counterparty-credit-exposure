"""
Computes counterparty exposure profiles from simulated MTM paths, with
and without a simple collateral agreement (CSA) applied.

Definitions used:
    Exposure at time t (per path)   = max(MTM_t, 0)
    Expected Exposure  EE(t)        = mean across paths of exposure
    Potential Future Exposure PFE(t) at 95% = 95th percentile across paths

Collateral (simplified one way CSA, posted by the counterparty to us):
    collateral_t = max(0, exposure_t_minus_1 - threshold)
    net_exposure_t = max(0, exposure_t - collateral_t)

Collateral is posted with a one step lag (a simple proxy for the margin
period of risk) and only reduces exposure down to the agreed threshold,
never below it. This is a simplification of real CSA mechanics (which
also include minimum transfer amounts, independent amounts and haircuts)
but it captures the core effect: collateral caps exposure, it does not
eliminate it.
"""

import numpy as np


def compute_exposure_profile(mtm_paths: np.ndarray) -> dict:
    exposure = np.maximum(mtm_paths, 0.0)
    ee = exposure.mean(axis=0)
    pfe_95 = np.percentile(exposure, 95, axis=0)
    return {"exposure_paths": exposure, "ee": ee, "pfe_95": pfe_95}


def apply_collateral(exposure_paths: np.ndarray, threshold: float) -> dict:
    n_paths, n_points = exposure_paths.shape
    net_exposure = np.empty_like(exposure_paths)
    net_exposure[:, 0] = np.maximum(exposure_paths[:, 0] - 0.0, 0.0)

    for t in range(1, n_points):
        collateral_posted = np.maximum(exposure_paths[:, t - 1] - threshold, 0.0)
        net_exposure[:, t] = np.maximum(exposure_paths[:, t] - collateral_posted, 0.0)

    ee_net = net_exposure.mean(axis=0)
    pfe_95_net = np.percentile(net_exposure, 95, axis=0)
    return {"net_exposure_paths": net_exposure, "ee_net": ee_net, "pfe_95_net": pfe_95_net}

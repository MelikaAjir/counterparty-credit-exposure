import numpy as np
import pytest

from exposure import apply_collateral, compute_exposure_profile


def test_exposure_never_negative():
    mtm = np.array([[-100.0, 50.0, -20.0], [30.0, -40.0, 10.0]])
    result = compute_exposure_profile(mtm)
    assert (result["exposure_paths"] >= 0).all()


def test_pfe_at_least_expected_exposure():
    rng = np.random.default_rng(0)
    mtm = rng.normal(0, 100, size=(1000, 5))
    result = compute_exposure_profile(mtm)
    assert (result["pfe_95"] >= result["ee"] - 1e-9).all()


def test_all_negative_mtm_gives_zero_exposure():
    mtm = -np.ones((10, 4)) * 500
    result = compute_exposure_profile(mtm)
    assert np.allclose(result["ee"], 0.0)
    assert np.allclose(result["pfe_95"], 0.0)


def test_collateral_reduces_exposure():
    rng = np.random.default_rng(1)
    mtm = rng.normal(500, 200, size=(2000, 10))
    base = compute_exposure_profile(mtm)
    collateralised = apply_collateral(base["exposure_paths"], threshold=100.0)
    # after the first period, collateralised expected exposure should be
    # no higher than uncollateralised expected exposure
    assert (collateralised["ee_net"][1:] <= base["ee"][1:] + 1e-9).all()


def test_collateral_never_negative():
    rng = np.random.default_rng(2)
    mtm = rng.normal(0, 300, size=(500, 8))
    base = compute_exposure_profile(mtm)
    collateralised = apply_collateral(base["exposure_paths"], threshold=50.0)
    assert (collateralised["net_exposure_paths"] >= 0).all()


def test_zero_threshold_collateral_caps_exposure_at_previous_step_zero():
    # with a zero threshold, all exposure above zero should in principle
    # be collateralised with a one step lag, so from step 1 onward net
    # exposure should be small relative to the raw exposure at the same
    # step when exposure is roughly stable
    exposure_paths = np.array([[100.0, 110.0, 105.0, 108.0]])
    collateralised = apply_collateral(exposure_paths, threshold=0.0)
    # first period cannot be collateralised (no prior exposure to post against)
    assert collateralised["net_exposure_paths"][0, 0] == 100.0
    # subsequent periods should be much closer to zero
    assert collateralised["net_exposure_paths"][0, 1] < 15.0

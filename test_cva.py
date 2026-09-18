import numpy as np
import pytest

from cva import compute_cva, hazard_survival_curve
from simulate_rates import discount_factors, simulate_vasicek_paths


def test_survival_curve_decreasing():
    time_grid = np.linspace(0, 5, 21)
    survival = hazard_survival_curve(hazard_rate=0.03, time_grid=time_grid)
    assert (np.diff(survival) <= 0).all()
    assert survival[0] == pytest.approx(1.0)


def test_cva_is_positive_for_positive_exposure():
    time_grid = np.linspace(0, 5, 21)
    ee = np.linspace(0, 100_000, 21)
    df = np.ones_like(time_grid)
    cva = compute_cva(ee, time_grid, df, hazard_rate=0.02, lgd=0.6)
    assert cva > 0


def test_cva_zero_when_exposure_zero():
    time_grid = np.linspace(0, 5, 21)
    ee = np.zeros(21)
    df = np.ones_like(time_grid)
    cva = compute_cva(ee, time_grid, df, hazard_rate=0.02, lgd=0.6)
    assert cva == pytest.approx(0.0)


def test_higher_hazard_rate_gives_higher_cva():
    time_grid = np.linspace(0, 5, 21)
    ee = np.linspace(0, 100_000, 21)
    df = np.ones_like(time_grid)
    low_h = compute_cva(ee, time_grid, df, hazard_rate=0.01, lgd=0.6)
    high_h = compute_cva(ee, time_grid, df, hazard_rate=0.05, lgd=0.6)
    assert high_h > low_h


def test_vasicek_paths_start_at_r0():
    paths = simulate_vasicek_paths(
        r0=0.03, a=0.3, b=0.03, sigma=0.01, n_paths=50, n_steps=10, dt=0.25
    )
    assert paths.shape == (50, 11)
    assert np.allclose(paths[:, 0], 0.03)


def test_vasicek_zero_volatility_is_deterministic():
    paths = simulate_vasicek_paths(
        r0=0.05, a=0.5, b=0.02, sigma=0.0, n_paths=20, n_steps=8, dt=0.25
    )
    # with zero volatility every path should be identical
    assert np.allclose(paths, paths[0])
    # and should mean revert towards b over time
    assert paths[0, -1] < paths[0, 0]


def test_discount_factors_decreasing_for_positive_rates():
    mean_rates = np.full(9, 0.03)
    df = discount_factors(mean_rates, dt=0.25)
    assert (np.diff(df) < 0).all()
    assert df[0] < 1.0

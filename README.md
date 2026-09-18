# Counterparty Credit Exposure & CVA Simulation

A small, honest quant project: a Monte Carlo interest rate simulation, a
counterparty exposure engine (Expected Exposure and 95% Potential Future
Exposure), a simple collateral (CSA) model, a CVA calculation, a rate
shock stress test, and a unit test suite. Built to demonstrate the
mechanics behind counterparty credit risk measurement, not to claim a
production grade pricing or risk engine.

## What this project actually does

- Simulates future short interest rate paths using a Vasicek mean
  reverting model (Monte Carlo, 5,000 paths).
- Values a single payer interest rate swap along each path, using a
  duration style approximation stated plainly in the code (see
  `portfolio.py`).
- Computes the counterparty exposure profile over time:
  - **Expected Exposure (EE)**: the average exposure across all
    simulated paths at each point in time.
  - **95% Potential Future Exposure (PFE)**: the 95th percentile of
    exposure at each point in time.
- Applies a simplified one way collateral agreement (a threshold based
  CSA, with collateral posted one period in arrears as a proxy for
  margin period of risk) and recomputes exposure net of collateral.
- Computes **CVA** (Credit Valuation Adjustment) from the expected
  exposure profile, using a constant hazard rate default model and a
  discount curve built from the simulated rates.
- Runs a **rate shock stress test** (a 150 basis point downward shock
  combined with higher volatility) and reports the resulting change in
  CVA.
- Is unit tested: 13 tests covering exposure non negativity, the PFE
  greater than or equal to EE relationship, collateral mechanics, CVA
  sensitivity to the hazard rate, and edge cases such as zero
  volatility and all negative mark to market paths.

## Honest result

Collateral meaningfully reduces both expected exposure and CVA in this
example (around 35 to 40 percent), which is the expected, textbook
direction of that effect. The stress scenario increases CVA
substantially, again in the expected direction. These results are
sanity checks that the mechanics behave the way counterparty risk
theory says they should, not a claim about any real counterparty or
real market conditions.

## Exposure Profile

![Exposure Profile](charts/exposure_profile.png)

## Repository Structure

```
├── simulate_rates.py   # Vasicek short rate Monte Carlo simulation
├── portfolio.py        # simplified payer swap mark to market
├── exposure.py         # EE, 95% PFE, and collateral (CSA) logic
├── cva.py              # CVA from an exposure profile and hazard curve
├── stress.py           # rate shock stress test
├── run_analysis.py     # ties everything together, prints results, saves chart
├── tests/
│   ├── test_exposure.py   # 6 tests: exposure and collateral mechanics
│   └── test_cva.py        # 7 tests: CVA, survival curve, rate simulation
├── charts/
├── data/
└── requirements.txt
```

## How to run

```
pip install -r requirements.txt
python run_analysis.py       # runs the full pipeline, prints results, saves chart
python -m pytest tests/ -v   # runs the 13 unit tests
```

## Known limitations (worth being upfront about)

- The swap mark to market is a first order, duration style
  approximation, not a full curve based valuation. It gets the sign and
  general shape of exposure right, but a real desk would value each leg
  against a bootstrapped discount curve.
- The collateral model captures the core effect of a CSA (exposure is
  capped near the threshold) but omits real world features such as
  minimum transfer amounts, independent amounts, and collateral
  haircuts.
- CVA uses a single flat hazard rate rather than a full credit curve
  bootstrapped from CDS spreads, and considers one trade rather than a
  netted counterparty portfolio.
- The stress test applies one hand chosen scenario. A real stress
  framework would run a full set of regulatory or historical scenarios.
- Vasicek rates can in principle go negative, which is a known
  limitation of the model; it is left as is here for simplicity rather
  than substituting a strictly positive short rate model.

## About

Built by Melika Ajir, MSci Mathematics, as a hands on project applying
Python (NumPy), Monte Carlo simulation, and counterparty credit risk
concepts (exposure, collateral, CVA, stress testing) relevant to a
counterparty and market risk context.


# Option Pricing Model — Portfolio-Ready Project

This repo gives you a clean, modular option pricing library + demo so you can:
- Price European options via **Black–Scholes**, **Binomial Tree (CRR)**, and **Monte Carlo (GBM)**.
- Compute key **Greeks**.
- Recover **implied volatility** and **calibrate** a constant-vol model to a small option chain.
- Extend to stochastic-volatility models (Heston/SABR): a scaffold is provided in `examples/` to plug in MCMC or least squares calibration.

## Install
```
pip install -r requirements.txt
```
*(only needs numpy for the example; core module uses Python stdlib to stay light)*

## Quickstart
```
from option_pricing import bs_price, bs_greeks, crr_price, mc_price_gbm, implied_vol_newton
S, K, T, r, q, sigma = 100, 100, 0.5, 0.03, 0.00, 0.2
print("BS call:", bs_price(S,K,T,r,sigma,q,"call"))
print("CRR call:", crr_price(S,K,T,r,sigma,200,q,"call"))
print("MC  call:", mc_price_gbm(S,K,T,r,sigma,q, n_paths=20000)["price"])
print("Greeks :", bs_greeks(S,K,T,r,sigma,q,"call"))
```

## What to include in your write-up
1. **Method overview:** what each model assumes; where it shines/fails.
2. **Numerical stability & convergence:** binomial step convergence; MC variance; CV/antithetic.
3. **Validation:** recover BS price with MC; unit tests vs. known values.
4. **Calibration:** show implied vols and a constant-vol fit; discuss errors and smile.
5. **Stretch:** implement Heston MC or the closed-form integral (Carr–Madan/Heston 1993) and calibrate with MCMC (you can reuse your MCMC sampler from coursework).

## Resume bullet (example)
- *Built a modular option pricing engine (BS, CRR, Monte Carlo) with Greeks and implied-vol calibration; validated convergence and error bounds; designed clean APIs and tests; packaged as a pip-installable module.*

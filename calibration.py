
import math
from typing import List, Tuple, Dict
from .implied_vol import implied_vol_newton
from .black_scholes import bs_price

def calibrate_const_vol(S, r, q, quotes: List[Tuple[float, float, float, float, str]]):
    """
    Calibrate a single constant vol by minimizing RMSE to given market prices.
    quotes: list of tuples (K, T, market_price, weight, option_type)
    Returns dict with sigma, rmse, details
    """
    # Start from implied vol of ATM-ish quote if available
    if not quotes:
        raise ValueError("Need at least one quote")
    import numpy as np

    # initial guess: median of individual implied vols
    ivs = []
    for (K, T, mkt, w, typ) in quotes:
        iv = implied_vol_newton(S, K, T, r, q, mkt, option=typ)
        if iv > 0:
            ivs.append(iv)
    sigma0 = float(np.median(ivs)) if ivs else 0.2

    def rmse(sigma):
        err2 = 0.0; wsum=0.0
        for (K, T, mkt, w, typ) in quotes:
            model = bs_price(S, K, T, r, sigma, q, option=typ)
            err2 += w * (model - mkt)**2
            wsum += w
        return math.sqrt(err2 / max(wsum, 1e-12))

    # 1D search (golden section)
    a, b = 1e-4, 3.0
    phi = (1 + 5 ** 0.5) / 2
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    for _ in range(100):
        if abs(c - d) < 1e-6:
            break
        if rmse(c) < rmse(d):
            b = d
        else:
            a = c
        c = b - (b - a) / phi
        d = a + (b - a) / phi

    sigma = 0.5 * (a + b)
    return {"sigma": sigma, "rmse": rmse(sigma)}

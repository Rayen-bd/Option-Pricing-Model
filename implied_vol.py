
import math

from .black_scholes import bs_price

def implied_vol_newton(S, K, T, r, q, option_price, option="call", tol=1e-8, max_iter=100):
    """
    Newton-Raphson search for implied vol.
    Returns sigma.
    """
    if option_price <= 0.0:
        return 0.0
    # Initial guess — Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2*math.pi/T) * option_price / S
    sigma = max(1e-8, min(5.0, sigma))

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, q, option)
        # Vega analytic
        if T <= 0: break
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        vega = (S * math.exp(-q*T) / math.sqrt(2*math.pi)) * math.exp(-0.5 * d1 * d1) * math.sqrt(T)
        diff = price - option_price
        if abs(diff) < tol:
            return sigma
        # Guard
        if vega < 1e-12:
            break
        sigma -= diff / vega
        # clamp
        sigma = max(1e-8, min(5.0, sigma))
    return sigma

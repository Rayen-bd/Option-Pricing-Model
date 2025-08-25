
import math

# Standard normal CDF and PDF
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _n_pdf(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def bs_price(S, K, T, r, sigma, q=0.0, option="call"):
    """
    Black–Scholes price for European call/put with continuous dividend yield q.
    Parameters: S (spot), K (strike), T (time in years), r (rf), sigma (vol), q (dividend yield)
    """
    if T <= 0:
        # immediate expiry payoff
        if option == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    if sigma <= 0:
        # zero vol degenerates to discounted intrinsic of forward
        F = S * math.exp((r - q) * T)
        if option == "call":
            return math.exp(-r*T) * max(F - K, 0.0)
        else:
            return math.exp(-r*T) * max(K - F, 0.0)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option.lower() == "call":
        return S * math.exp(-q*T) * _phi(d1) - K * math.exp(-r*T) * _phi(d2)
    else:
        return K * math.exp(-r*T) * _phi(-d2) - S * math.exp(-q*T) * _phi(-d1)

def bs_greeks(S, K, T, r, sigma, q=0.0, option="call"):
    """
    Returns a dict of greeks: delta, gamma, vega (per 1% = 0.01 vol), theta (per year), rho (per 1%)
    """
    if T <= 0 or sigma <= 0:
        # Use finite differences for the edge case to avoid division by zero.
        eps = 1e-5
        base = bs_price(S, K, T, r, sigma, q, option)
        dS = bs_price(S+eps, K, T, r, sigma, q, option) - bs_price(S-eps, K, T, r, sigma, q, option)
        delta = dS / (2*eps)
        gamma = (bs_price(S+eps, K, T, r, sigma, q, option) - 2*base + bs_price(S-eps, K, T, r, sigma, q, option)) / (eps*eps)
        vega = (bs_price(S, K, T, r, sigma+eps, q, option) - base) / eps / 100.0
        theta = (bs_price(S, K, T-eps, r, sigma, q, option) - base) / (-eps)  # per year
        rho = (bs_price(S, K, T, r+eps, sigma, q, option) - base) / eps / 100.0
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = _phi(d1)
    nd1 = _n_pdf(d1)

    if option.lower() == "call":
        delta = math.exp(-q*T) * Nd1
        rho =  T * K * math.exp(-r*T) * _phi(d2) / 100.0
        theta = (- (S * math.exp(-q*T) * nd1 * sigma) / (2.0 * math.sqrt(T))
                 - r * K * math.exp(-r*T) * _phi(d2)
                 + q * S * math.exp(-q*T) * Nd1)
    else:
        delta = math.exp(-q*T) * (Nd1 - 1.0)
        rho = -T * K * math.exp(-r*T) * _phi(-d2) / 100.0
        theta = (- (S * math.exp(-q*T) * nd1 * sigma) / (2.0 * math.sqrt(T))
                 + r * K * math.exp(-r*T) * _phi(-d2)
                 - q * S * math.exp(-q*T) * _phi(-d1))

    gamma = (math.exp(-q*T) * nd1) / (S * sigma * math.sqrt(T))
    vega = (S * math.exp(-q*T) * nd1 * math.sqrt(T)) / 100.0  # per 1%

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

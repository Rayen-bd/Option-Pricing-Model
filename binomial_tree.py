
import math

def crr_price(S, K, T, r, sigma, N=100, q=0.0, option="call", american=False):
    """
    Cox-Ross-Rubinstein binomial tree for European/optional American options with dividend yield q.
    Returns price.
    """
    if N < 1:
        N = 1
    dt = T / N
    if dt <= 0:
        return max(0.0, (S-K) if option=="call" else (K-S))

    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)
    if p < 0.0 or p > 1.0:
        # Guard against arbitrage region numerics; clamp
        p = max(0.0, min(1.0, p))

    # Initialize terminal node payoffs
    prices = [0.0] * (N + 1)
    for i in range(N + 1):
        S_T = S * (u ** i) * (d ** (N - i))
        prices[i] = max(0.0, (S_T - K) if option=="call" else (K - S_T))

    # Backward induction
    disc = math.exp(-r * dt)
    for n in range(N - 1, -1, -1):
        for i in range(n + 1):
            cont = disc * (p * prices[i+1] + (1.0 - p) * prices[i])
            if american:
                S_n = S * (u ** i) * (d ** (n - i))
                intrinsic = max(0.0, (S_n - K) if option=="call" else (K - S_n))
                prices[i] = max(cont, intrinsic)
            else:
                prices[i] = cont
    return prices[0]


import math
import random
from statistics import pstdev

def heston_mc_price(S, K, T, r, q, kappa, theta, sigma_v, rho, v0,
                    n_paths=20000, n_steps=100, antithetic=True, seed=123,
                    scheme="full_truncation", option="call"):
    """
    Monte Carlo pricing for European options under the Heston model using
    log-Euler for S and full-truncation Euler for v (default).
    dS_t/S_t = (r - q) dt + sqrt(v_t) dW1_t
    dv_t     = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW2_t
    corr(dW1, dW2) = rho

    Returns dict with price and stderr.
    """
    if T <= 0.0:
        payoff = max(0.0, S-K) if option=="call" else max(0.0, K-S)
        return {"price": math.exp(-r*T) * payoff, "stderr": 0.0}

    disc = math.exp(-r*T)
    dt = T / float(n_steps)
    sqrt_dt = math.sqrt(dt)
    rng = random.Random(seed)

    payoffs = []
    n_pairs = n_paths // (2 if antithetic else 1)

    for _ in range(n_pairs):
        z1 = rng.gauss(0.0, 1.0); z1b = -z1 if antithetic else rng.gauss(0.0, 1.0)
        z2 = rng.gauss(0.0, 1.0); z2b = -z2 if antithetic else rng.gauss(0.0, 1.0)

        # Construct correlated normals
        def step_pair(z1_, z2_):
            # correlation via Cholesky
            w1 = z1_
            w2 = rho * z1_ + math.sqrt(max(1.0 - rho*rho, 0.0)) * z2_
            return w1, w2

        for (zz1, zz2) in ((z1, z2), (z1b, z2b)) if antithetic else ((z1, z2),):
            S_t = S
            v_t = max(v0, 1e-12)
            for _n in range(n_steps):
                w1, w2 = step_pair(rng.gauss(0.0,1.0), rng.gauss(0.0,1.0))  # new shocks each step
                # full truncation (Andersen 2008)
                v_pos = max(v_t, 0.0)
                if scheme == "full_truncation":
                    v_next = v_t + kappa*(theta - v_pos)*dt + sigma_v*math.sqrt(v_pos)*w2*sqrt_dt
                    v_next = max(v_next, 0.0)
                else:
                    # plain Euler with floor
                    v_next = v_t + kappa*(theta - v_t)*dt + sigma_v*math.sqrt(v_pos)*w2*sqrt_dt
                    v_next = max(v_next, 0.0)

                # Log-Euler for S to keep positivity
                S_t = S_t * math.exp((r - q - 0.5*v_pos)*dt + math.sqrt(v_pos)*w1*sqrt_dt)
                v_t = v_next

            payoff = max(0.0, S_t - K) if option=="call" else max(0.0, K - S_t)
            payoffs.append(disc * payoff)

    if len(payoffs) == 0:
        return {"price": 0.0, "stderr": 0.0}

    mean = sum(payoffs) / len(payoffs)
    std = pstdev(payoffs) if len(payoffs) > 1 else 0.0
    return {"price": mean, "stderr": std / math.sqrt(len(payoffs))}

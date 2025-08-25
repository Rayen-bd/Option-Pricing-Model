
"""
Lightweight Metropolis–Hastings calibration for the Heston model.
- Assumes independent Gaussian errors for observed option prices.
- Uses random-walk proposals with reflection/clipping to maintain constraints.
- Intended as a scaffold: keep iteration counts small in examples; tune for real use.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable

from .heston_mc import heston_mc_price

@dataclass
class Quote:
    K: float
    T: float
    price: float
    option: str  # "call" or "put"
    weight: float = 1.0

def log_prior(params: Dict[str, float]) -> float:
    """
    Weakly-informative priors (log densities) that keep parameters in sane ranges.
    - kappa, theta, sigma_v, v0: half-normal (sd=2.0)
    - rho: uniform(-0.999, 0.999)
    """
    kappa = params["kappa"]; theta = params["theta"]; sigma_v = params["sigma_v"]
    rho = params["rho"]; v0 = params["v0"]
    if not (-0.999 < rho < 0.999): return -math.inf
    for x in [kappa, theta, sigma_v, v0]:
        if x <= 0.0 or not math.isfinite(x):
            return -math.inf
    sd = 2.0
    log_halfnorm = -math.log(sd*math.sqrt(2*math.pi)) - 0.5*(x/sd)**2  # placeholder
    # compute properly per parameter
    lp = 0.0
    for x in [kappa, theta, sigma_v, v0]:
        lp += -math.log(sd*math.sqrt(2*math.pi)) - 0.5*(x/sd)**2 + math.log(2.0)  # half-normal on x>0
    lp += -math.log(1.998)  # uniform for rho in (-0.999,0.999)
    return lp

def log_likelihood(S: float, r: float, q: float, quotes: List[Quote],
                   params: Dict[str, float],
                   pricer: Callable[..., Dict[str,float]] = heston_mc_price,
                   n_paths:int=5000, n_steps:int=50, obs_sd: float=0.05) -> float:
    """
    Gaussian errors with sd proportional to option moneyness (obs_sd * max(1, S*0.01)).
    For 'obs_sd', 0.05 corresponds roughly to 5 cents error-scale.
    """
    ll = 0.0
    for qte in quotes:
        model = pricer(S, qte.K, qte.T, r, q, params["kappa"], params["theta"],
                       params["sigma_v"], params["rho"], params["v0"],
                       n_paths=n_paths, n_steps=n_steps, option=qte.option)["price"]
        sd = max(obs_sd, 0.01 * (1.0 + qte.K/ max(S,1e-6)))  # crude scale
        resid = model - qte.price
        ll += qte.weight * (-0.5 * (resid/sd)**2 - math.log(sd*math.sqrt(2*math.pi)))
    return ll

def propose(params: Dict[str,float], step: Dict[str,float], rng: random.Random) -> Dict[str,float]:
    """
    Gaussian random-walk with reflection/clipping.
    """
    out = dict(params)
    for k in ["kappa","theta","sigma_v","v0"]:
        out[k] = abs(out[k] + rng.gauss(0.0, step[k]))  # reflect at 0
        out[k] = max(out[k], 1e-6)
    out["rho"] = max(-0.999, min(0.999, out["rho"] + rng.gauss(0.0, step["rho"])))
    return out

def mh_sample(S: float, r: float, q: float, quotes: List[Quote],
              init: Dict[str,float], step: Dict[str,float],
              n_iter:int=1000, burn:int=200, thin:int=1,
              seed:int=123, obs_sd:float=0.05,
              n_paths:int=3000, n_steps:int=40) -> Dict[str,object]:
    """
    Run a basic MH sampler and return draws and acceptance stats.
    """
    rng = random.Random(seed)
    curr = dict(init)
    curr_lp = log_prior(curr) + log_likelihood(S, r, q, quotes, curr, n_paths=n_paths, n_steps=n_steps, obs_sd=obs_sd)
    draws = []
    acc = 0
    for it in range(n_iter):
        prop = propose(curr, step, rng)
        prop_lp = log_prior(prop) + log_likelihood(S, r, q, quotes, prop, n_paths=n_paths, n_steps=n_steps, obs_sd=obs_sd)
        if math.log(rng.random()) < (prop_lp - curr_lp):
            curr, curr_lp = prop, prop_lp
            acc += 1
        if it >= burn and ((it - burn) % thin == 0):
            draws.append(dict(curr))
    return {
        "draws": draws,
        "accept_rate": acc / max(n_iter,1),
    }

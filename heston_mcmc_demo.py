
from option_pricing import heston_mc_price, mh_sample, Quote, bs_price

# Synthetic setup
S, r, q = 100.0, 0.01, 0.0
true = dict(kappa=2.0, theta=0.04, sigma_v=0.6, rho=-0.6, v0=0.04)

# Build a small option "chain" at one maturity
T = 0.5
strikes = [80, 90, 100, 110, 120]
quotes = []
for K in strikes:
    # Use Heston MC to generate "market" price (could add noise for realism)
    mkt = heston_mc_price(S, K, T, r, q, **true, n_paths=20000, n_steps=100)["price"]
    quotes.append(Quote(K=K, T=T, price=mkt, option="call", weight=1.0))

# Initial guess + steps
init = dict(kappa=1.0, theta=0.03, sigma_v=0.4, rho=-0.3, v0=0.03)
step = dict(kappa=0.3, theta=0.02, sigma_v=0.2, rho=0.1, v0=0.02)

# Run a short MH chain (keep small for demo)
res = mh_sample(S, r, q, quotes, init, step, n_iter=400, burn=100, thin=2, n_paths=5000, n_steps=40)

# Summarize
import statistics as stats
def summarize(draws, key):
    vals = [d[key] for d in draws]
    return dict(mean=stats.mean(vals), median=stats.median(vals),
                p05=sorted(vals)[int(0.05*len(vals))],
                p95=sorted(vals)[int(0.95*len(vals))-1])

print("Acceptance rate:", round(res["accept_rate"], 3))
for k in true.keys():
    s = summarize(res["draws"], k)
    print(f"{k:8s} true={true[k]:.3f}  mean={s['mean']:.3f}  5-95% [{s['p05']:.3f},{s['p95']:.3f}]")

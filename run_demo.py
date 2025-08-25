
from option_pricing import bs_price, bs_greeks, crr_price, mc_price_gbm, implied_vol_newton
from option_pricing.calibration import calibrate_const_vol

S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.03, 0.0, 0.2

print("=== Black–Scholes ===")
print("call:", bs_price(S,K,T,r,sigma,q,"call"))
print("put :", bs_price(S,K,T,r,sigma,q,"put"))
print("greeks(call):", bs_greeks(S,K,T,r,sigma,q,"call"))

print("\n=== Binomial (CRR) ===")
print("call:", crr_price(S,K,T,r,sigma,200,q,"call"))
print("put :", crr_price(S,K,T,r,sigma,200,q,"put"))

print("\n=== Monte Carlo (GBM) ===")
mc = mc_price_gbm(S,K,T,r,sigma,q, n_paths=50000)
print("call (MC):", mc["price"], "stderr:", mc["stderr"])

print("\n=== Implied Vol & Calibration ===")
mkt_call = bs_price(S,K,T,r,0.24,q,"call")  # pretend 'market'
iv = implied_vol_newton(S,K,T,r,q,mkt_call,"call")
print("Implied vol from price:", round(iv,6))

quotes = [
    (90.0, 0.5, bs_price(S,90.0,T,r,0.24,q,"call"), 1.0, "call"),
    (100.0,0.5, bs_price(S,100.0,T,r,0.24,q,"call"), 1.0, "call"),
    (110.0,0.5, bs_price(S,110.0,T,r,0.24,q,"call"), 1.0, "call"),
]
fit = calibrate_const_vol(S, r, q, quotes)
print("Const-vol calibration:", fit)

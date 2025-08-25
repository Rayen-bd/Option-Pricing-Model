
from option_pricing import bs_price, crr_price, mc_price_gbm

def test_bs_put_call_parity():
    S,K,T,r,q,sigma = 100,100,1.0,0.05,0.0,0.2
    call = bs_price(S,K,T,r,sigma,q,"call")
    put  = bs_price(S,K,T,r,sigma,q,"put")
    lhs = call - put
    rhs = S*pow(2.718281828459045, -q*T) - K*pow(2.718281828459045, -r*T)
    assert abs(lhs - rhs) < 1e-8

def test_binomial_converges_to_bs():
    S,K,T,r,q,sigma = 100,100,0.5,0.03,0.0,0.2
    bs = bs_price(S,K,T,r,sigma,q,"call")
    b = crr_price(S,K,T,r,sigma, N=1000, q=q, option="call")
    assert abs(b - bs) / bs < 5e-3  # within 0.5%

def test_mc_close_to_bs():
    S,K,T,r,q,sigma = 100,100,0.2,0.01,0.0,0.3
    bs = bs_price(S,K,T,r,sigma,q,"call")
    mc = mc_price_gbm(S,K,T,r,sigma,q, n_paths=20000)["price"]
    assert abs(mc - bs) / bs < 0.03  # within 3%

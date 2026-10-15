"""512/768-bit energy nodes; no binary64 input enters a kernel."""
from functools import lru_cache
import gmpy2 as g
M=g.mpfr

def legendre(n,x):
    p0,p1=M(1),x
    if n==0:return p0,M(0)
    for k in range(2,n+1):
        p0,p1=p1,((2*k-1)*x*p1-(k-1)*p0)/k
    return p1,n*(x*p1-p0)/(x*x-1)

@lru_cache(maxsize=6)
def gauss_nodes(n,bits):
    positive=[]
    tol=g.exp2(-bits+24)
    for j in range(1,n//2+1):
        x=g.cos(g.const_pi()*(M(j)-M('0.25'))/(M(n)+M('0.5')))
        for iteration in range(40):
            p,dp=legendre(n,x);step=p/dp;x-=step
            if abs(step)<tol:break
        else:raise ArithmeticError('Gauss-Legendre root did not converge')
        positive.append(x)
    return tuple(sorted([-x for x in positive]+([M(0)]if n%2 else[])+positive))

def energy_nodes(cfg):
    n=int(cfg['n_mu']);family=cfg.get('energy_grid','chebyshev_midpoint')
    if family=='chebyshev_midpoint':
        return [(1-g.cos(M(2*p-1)*g.const_pi()/(2*n)))/2 for p in range(1,n+1)]
    if family=='chebyshev_open':
        z=[(1-g.cos(M(j)*g.const_pi()/n))/2 for j in range(1,n+1)]
        z[-1]=M(1)
        if n%2==0:z[n//2-1]=M('0.5')
        return z
    if family=='gauss_legendre_gap':
        return [(x+1)/2 for x in gauss_nodes(n-1,g.get_context().precision)]+[M(1)]
    raise ValueError('Unknown unbounded-domain grid '+family)

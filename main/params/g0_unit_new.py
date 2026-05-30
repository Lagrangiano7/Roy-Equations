"""G0 wave"""

import matplotlib.pyplot as plt
import numpy as np
import cmath as c
from numpy.linalg import norm


mpi=0.13957
sm=1.4**2
s0=1.75**2
alpha=0.5
s_ine=(1.05)**2

N=2

par= 0.248,  2.030,  4.62690459, 0.170
err_par=0.080, 0.045/N, 0.2, 0.030

def q(s,m):
    return c.sqrt((s/4)-m**2)

def sigma(s,m):
    if s.imag>=0:
        return c.sqrt(1-(4*m**2/s))
    else:
        return -c.sqrt(1-(4*m**2/s))
    
"""B-W"""

def q0(mg):
    return q(mg**2,mpi)

def D(s):
    return 11025 + 1575*s**2 + 135*s**4 + 10*s**6 + s**8

def G(s,Gg,mg,Rg):
    return Gg*(q(s,mpi)/q0(mg))**9*D(q0(mg)*Rg)/D(q(s,mpi)*Rg)

def tbw(s,Gg,mg,Rg,xg):
    return xg*mg*G(s,Gg,mg,Rg)/(sigma(s,mpi)*(mg**2-s-1j*mg*G(s,Gg,mg,Rg)))

def Sbw(s,Gg,mg,Rg,xg):
    return 1+ 2j*sigma(s,mpi)*tbw(s,Gg,mg,Rg,xg)

def etabw(s,Gg,mg,Rg,xg):
    return abs(Sbw(s,Gg,mg,Rg,xg))

def deltabw(s,Gg,mg,Rg,xg):
    return 90*c.phase(Sbw(s,Gg,mg,Rg,xg))/np.pi

"""Conformal"""

def v(s):
    return (c.sqrt(s)-alpha*c.sqrt(s0-s))/(c.sqrt(s)+alpha*c.sqrt(s0-s))

def w(s):
    return v(s+1j*10**(-18))

# --------- Matching structures---------------
def fac(s):
    return c.sqrt(s)/(2*q(s,mpi)**9)

def cotdbw_eff(s,Gg,mg,Rg,xg):
    return 1/(np.tan(c.phase(Sbw(s,Gg,mg,Rg,xg)).real/2)*fac(s))

def derC(Gg,mg,Rg,xg):
    return (cotdbw_eff(sm+10**(-5),Gg,mg,Rg,xg)-cotdbw_eff(sm-10**(-5),Gg,mg,Rg,xg))/(2*10**(-5))

derW=(w(sm+10**(-5))-w(sm-10**(-5)))/(2*10**(-5))

def b1(Gg,mg,Rg,xg):
    return derC(Gg,mg,Rg,xg)/derW

def b0(Gg,mg,Rg,xg):
    return cotdbw_eff(sm,Gg,mg,Rg,xg) - b1(Gg,mg,Rg,xg)*w(sm)

#  -------------------------------------------------

def phi(s,Gg,mg,Rg,xg):
    return (b0(Gg,mg,Rg,xg) + b1(Gg,mg,Rg,xg)*w(s))/(sigma(s,mpi)*q(s,mpi)**8)

def deltac(s,Gg,mg,Rg,xg):
    return 180*np.arctan(1/phi(s,Gg,mg,Rg,xg))/np.pi

def delta(s,Gg,mg,Rg,xg):
    if s<sm:
        return deltac(s,Gg,mg,Rg,xg).real
    else:
        return deltabw(s,Gg,mg,Rg,xg).real

# --------- Matching structures---------------
    
def eps(Gg,mg,Rg,xg):
    return (1-etabw(sm,Gg,mg,Rg,xg))/(1-(s_ine/sm))**(9/2)

def derE(Gg,mg,Rg,xg):
    return (etabw(sm+10**(-5),Gg,mg,Rg,xg)-etabw(sm-10**(-5),Gg,mg,Rg,xg))/(2*10**(-5))

def r(Gg,mg,Rg,xg):
    return -(sm*derE(Gg,mg,Rg,xg)/(1-etabw(sm,Gg,mg,Rg,xg)) +(9/2)*s_ine/(sm-s_ine))

#  -------------------------------------------------
    
def eta(s,Gg,mg,Rg,xg):
    if s<s_ine:
        return 1
    elif s<sm:
        return 1-eps(Gg,mg,Rg,xg).real*(1-(s_ine/s))**(9/2)*(1+r(Gg,mg,Rg,xg).real*(1-(sm/s)))
    else:
        return etabw(s,Gg,mg,Rg,xg).real

def tf(s,params):
    Gg,mg,Rg,xg=params
    return (eta(s,Gg,mg,Rg,xg)*c.exp(np.pi*1j*delta(s,Gg,mg,Rg,xg)/90)-1)/(2j*sigma(s,mpi))

def deltafr(s,params):
    Gg,mg,Rg,xg=params
    return delta(s,Gg,mg,Rg,xg)

def etafr(s,params):
    Gg,mg,Rg,xg=params
    return eta(s,Gg,mg,Rg,xg)
    
def erru(f,s,parf,err_parf):
    n=len(parf)
    r=np.zeros(n)
    fv=f(s,parf)
    for i in range(0,n):
        par1=list(parf)
        par2=list(parf)
        par1[i]=par1[i]+err_parf[i]
        par2[i]=par2[i]-err_parf[i]
        w1=f(s,par1)-fv
        w2=f(s,par2)-fv
        m=max(w1,w2)
        if w1*w2<0:
            r[i]=m
        elif m>0:
            r[i]=m
    return norm(r)

def errd(f,s,parf,err_parf):
    n=len(parf)
    r=np.zeros(n)
    fv=f(s,parf)
    for i in range(0,n):
        par1=list(parf)
        par2=list(parf)
        par1[i]=par1[i]+err_parf[i]
        par2[i]=par2[i]-err_parf[i]
        w1=f(s,par1)-fv
        w2=f(s,par2)-fv
        m=min(w1,w2)
        if w1*w2<0:
            r[i]=m
        elif m<0:
            r[i]=m
    return - norm(r)

def errdu(s,parf,err_parf):
    return erru(deltafr,s,parf,err_parf)
def errdd(s,parf,err_parf):
    return errd(deltafr,s,parf,err_parf)

def erreu(s,parf,err_parf):
    return erru(etafr,s,parf,err_parf)
def erred(s,parf,err_parf):
    return errd(etafr,s,parf,err_parf)

# Deleted all the paper stuff
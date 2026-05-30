"""F wave Solution II"""

import numpy as np
import cmath as c
from numpy.linalg import norm

mpi=0.13957
sm=1.2**2
s0=1.5**2
alpha=0.5
mpi0=0.13957
mom=0.78266
s_ine=(mpi0+mom)**2

N=2

par=0.4075659 , -0.91384224,  0.26424642,  1.72532612,  4.92516723,        0.24954742
err_par=0.02395786/N,0.05834034,0.01302545/N, 0.00408748/N, 0.02778393/N, 0.0178626/N

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
    return 225 + 45*s**2 + 6*s**4 + s**6

def G(s,Gg,mg,Rg):
    return Gg*(q(s,mpi)/q0(mg))**7*D(q0(mg)*Rg)/D(q(s,mpi)*Rg)

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
    return c.sqrt(s)/(2*q(s,mpi)**7)

def cotdbw_eff(s,Gg,mg,Rg,xg):
    return 1/(np.tan(c.phase(Sbw(s,Gg,mg,Rg,xg)).real/2)*fac(s))

def derC(Gg,mg,Rg,xg):
    return (cotdbw_eff(sm+10**(-5),Gg,mg,Rg,xg)-cotdbw_eff(sm-10**(-5),Gg,mg,Rg,xg))/(2*10**(-5))

derW=(w(sm+10**(-5))-w(sm-10**(-5)))/(2*10**(-5))

def b1(b2,b3,Gg,mg,Rg,xg):
    return derC(Gg,mg,Rg,xg)/derW - 2*b2*w(sm) - 3*b3*w(sm)**2

def b0(b2,b3,Gg,mg,Rg,xg):
    return cotdbw_eff(sm,Gg,mg,Rg,xg) - b1(b2,b3,Gg,mg,Rg,xg)*w(sm)- b2*w(sm)**2 - b3*w(sm)**3

#  -------------------------------------------------

def phi(s,b2,b3,Gg,mg,Rg,xg):
    return (b0(b2,b3,Gg,mg,Rg,xg) + b1(b2,b3,Gg,mg,Rg,xg)*w(s)+ b2*w(s)**2 + b3*w(s)**3)/(sigma(s,mpi)*q(s,mpi)**6)

def deltac(s,b2,b3,Gg,mg,Rg,xg):
    return 180*np.arctan(1/phi(s,b2,b3,Gg,mg,Rg,xg))/np.pi

def delta(s,b2,b3,Gg,mg,Rg,xg):
    if s<sm:
        return deltac(s,b2,b3,Gg,mg,Rg,xg).real
    else:
        return deltabw(s,Gg,mg,Rg,xg).real

# --------- Matching structures---------------
    
def eps(Gg,mg,Rg,xg):
    return (1-etabw(sm,Gg,mg,Rg,xg))/(1-(s_ine/sm))**(7/2)

def derE(Gg,mg,Rg,xg):
    return (etabw(sm+10**(-5),Gg,mg,Rg,xg)-etabw(sm-10**(-5),Gg,mg,Rg,xg))/(2*10**(-5))

def r(Gg,mg,Rg,xg):
    return -(sm*derE(Gg,mg,Rg,xg)/(1-etabw(sm,Gg,mg,Rg,xg)) +(7/2)*s_ine/(sm-s_ine))

#  -------------------------------------------------
    
def eta(s,Gg,mg,Rg,xg):
    if s<s_ine:
        return 1
    elif s<sm:
        return 1-eps(Gg,mg,Rg,xg).real*(1-(s_ine/s))**(7/2)*(1+r(Gg,mg,Rg,xg).real*(1-(sm/s)))
    else:
        return etabw(s,Gg,mg,Rg,xg).real

def tf(s,params):
    b2,b3,Gg,mg,Rg,xg=params
    return (eta(s,Gg,mg,Rg,xg)*c.exp(np.pi*1j*delta(s,b2,b3,Gg,mg,Rg,xg)/90)-1)/(2j*sigma(s,mpi))

def deltaf(s,params):
    b2,b3,Gg,mg,Rg,xg=params
    return delta(s,b2,b3,Gg,mg,Rg,xg)

def etaf(s,params):
    b2,b3,Gg,mg,Rg,xg=params
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
    return erru(deltaf,s,parf,err_parf)
def errdd(s,parf,err_parf):
    return errd(deltaf,s,parf,err_parf)

def erreu(s,parf,err_parf):
    return erru(etaf,s,parf,err_parf)
def erred(s,parf,err_parf):
    return errd(etaf,s,parf,err_parf)
"""New S2 Solution"""

import numpy as np
import cmath as c
from numpy.linalg import norm
from scipy.optimize import minimize, fmin, fsolve


# UNDER 0.85 GeV

mpi=0.13957
s0=1.05**2
sh=2.3**2
eps=0.17015

par=-7.92129719e+01, -6.80500205e+01,  1.49862692e-01,  4.23228992e+02,       -2.18692701e+03,  2.31908350e-01
err_par=2.8,10.5,0.0039249,109.14762483,355.39425489,0.06171356

def sigma(s,m):
    if s.imag>=0:
        return c.sqrt(1-(4*m**2/s))
    else:
        return -c.sqrt(1-(4*m**2/s))

def q(s,m):
    return c.sqrt((s/4)-m**2)

def v(s):
    return (c.sqrt(s)-c.sqrt(s0-s))/(c.sqrt(s)+c.sqrt(s0-s))

def w(s):
    return v(s+1j*10**(-18))

def v2(s):
    return (c.sqrt(s)-c.sqrt(sh-s))/(c.sqrt(s)+c.sqrt(sh-s))

def w2(s):
    return v2(s+1j*10**(-18))

wm=w2(0.85**2)

def Bh0(b0,b1):
    return b0 + b1*w(0.85**2)

def Bh1(b1):
    return b1*(s0/sh)*(np.sqrt(sh-0.85**2)/np.sqrt(s0-0.85**2))*((0.85+np.sqrt(sh-0.85**2))/(0.85+np.sqrt(s0-0.85**2)))**2

def phi20(s,b0,b1,z2,Bh2,Bh3):
    if s<0.85**2:
        return c.sqrt(s)*mpi**2*(b0 + b1*w(s))/(2*q(s,mpi)*(s-2*z2**2))
    else:
        return c.sqrt(s)*mpi**2*(Bh0(b0,b1) + Bh1(b1)*(w2(s)-wm) + Bh2*(w2(s)-wm)**2 + Bh3*(w2(s)-wm)**3)/(2*q(s,mpi)*(s-2*z2**2))

def deltaf(s,b0,b1,z2,Bh2,Bh3):
    return 180*np.arctan(1/phi20(s,b0,b1,z2,Bh2,Bh3)).real/np.pi

def etaf(s,eps):
    if s<0.915**2:
        return 1
    else:
        return 1-eps*(1-(0.915**2/s))**(3/2)


def tf(s,params):
    b0,b1,z2,Bh2,Bh3,eps=params
    return (etaf(s,eps)*c.exp(np.pi*1j*deltaf(s,b0,b1,z2,Bh2,Bh3)/90)-1)/(2j*sigma(s,mpi))

def S(s,params):
    return 1+ 2j*sigma(s,mpi)*tf(s,params)

def deltaf2(s,params):
    return 90*c.phase(S(s,params))/np.pi

def etafr(s,params):
    b0,b1,z2,Bh2,Bh3,eps=params
    return  etaf(s,eps)

"""Errors"""

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
    return erru(deltaf2,s,parf,err_parf)
def errdd(s,parf,err_parf):
    return errd(deltaf2,s,parf,err_parf)

def erreu(s,parf,err_parf):
    return erru(etafr,s,parf,err_parf)
def erred(s,parf,err_parf):
    return errd(etafr,s,parf,err_parf)
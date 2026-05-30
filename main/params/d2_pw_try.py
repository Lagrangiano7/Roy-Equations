"""New D2"""

import numpy as np
import cmath as c
from numpy.linalg import norm


# UNDER 0.85 GeV

mpi=0.13957
s0=1.45**2
sh=2.4**2
eps=0.000164

N1=4
N2=5
N3=2
N=2.3

par=1.09079014,  0.05376754,  5.72429053, 25.68267149, 15.43069636,        0.23159591
err_par=N1*0.04817726,N2*0.23977739,N3*0.58478895,N*2.17907351,N*7.55403152,N*0.00607046


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

def Bh0(b0,b1,b2):
    return b0 + b1*w(0.85**2) + b2*w(0.85**2)**2

def Bh1(b1,b2):
    return (b1+2*b2*w(0.85**2))*(s0/sh)*(np.sqrt(sh-0.85**2)/np.sqrt(s0-0.85**2))*((0.85+np.sqrt(sh-0.85**2))/(0.85+np.sqrt(s0-0.85**2)))**2

def phi22(s,b0,b1,b2,Bh2,Bh3,Delta):
    if s<0.85**2:
        return c.sqrt(s)*s*(b0 + b1*w(s) + b2*w(s)**2)/(2*q(s,mpi)**5*(4*(mpi**2 + Delta**2)-s))
    else:
        return c.sqrt(s)*s*(Bh0(b0,b1,b2) + Bh1(b1,b2)*(w2(s)-wm) + Bh2*(w2(s)-wm)**2+ Bh3*(w2(s)-wm)**3)/(2*q(s,mpi)**5*(4*(mpi**2 + Delta**2)-s))

def deltaf(s,b0,b1,b2,Bh2,Bh3,Delta):
    return 180*np.arctan(1/phi22(s,b0,b1,b2,Bh2,Bh3,Delta)).real/np.pi

def tf(s,params):
    b0,b1,b2,Bh2,Bh3,Delta=params
    return (c.exp(np.pi*1j*deltaf(s,b0,b1,b2,Bh2,Bh3,Delta)/90)-1)/(2j*sigma(s,mpi))

def S(s,params):
    return 1+ 2j*sigma(s,mpi)*tf(s,params)

def etafr(s,params):
    return abs(S(s,params))

def deltaf2(s,params):
    b0,b1,b2,Bh2,Bh3,Delta=params
    return deltaf(s,b0,b1,b2,Bh2,Bh3,Delta)

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
"""New G2 Solution"""

import numpy as np
import cmath as c
from numpy.linalg import norm

# UNDER 0.85 GeV

mpi=0.13957
s0=1.65**2
sh=2.4**2
eps=0.000164
sm=0.85**2

N=1
N2=1

par= 5.85425392e-01,  9.18173557e-01,  4.27475547e+00,  1.85907026e+02,        2.72930332e-01
err_par=N*0.00044026,N*0.00070194,N2*7.84247897,N2*19.45886503,0.00599783

N1=6
N2=7
N3=7
par=-0.66354937,  -8.95001319, -11.08692289,  19.69402062,         8.31341559,  2.72930332e-01
err_par=0.08104828/N1,0.1333764/N2,0.18907597/N3,1.48926165,2.96469609,0.05624994*0.4

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

def phi42(s,b0,b1,b2,Bh2,Bh3,Delta):
    if s<0.85**2:
        return c.sqrt(s)*s*(b0 + b1*w(s) + b2*w(s)**2)/(2*q(s,mpi)**9*(4*(mpi**2 + Delta**2)-s))
    else:
        return c.sqrt(s)*s*(Bh0(b0,b1,b2) + Bh1(b1,b2)*(w2(s)-wm) + Bh2*(w2(s)-wm)**2+ Bh3*(w2(s)-wm)**3)/(2*q(s,mpi)**9*(4*(mpi**2 + Delta**2)-s))

def deltaf(s,b0,b1,b2,Bh2,Bh3,Delta):
    return 180*np.arctan(1/phi42(s,b0,b1,b2,Bh2,Bh3,Delta)).real/np.pi


def tf(s,params):
    b0,b1,b2,Bh2,Bh3,Delta=params
    return (c.exp(np.pi*1j*deltaf(s,b0,b1,b2,Bh2,Bh3,Delta)/90)-1)/(2j*sigma(s,mpi))


def S(s,params):
    return 1+ 2j*sigma(s,mpi)*tf(s,params)

def etaf2(s,params):
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
    return erru(etaf2,s,parf,err_parf)
def erred(s,parf,err_parf):
    return errd(etaf2,s,parf,err_parf)
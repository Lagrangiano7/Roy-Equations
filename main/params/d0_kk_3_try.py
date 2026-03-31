"""CFD D0 Solution II"""

import numpy as np
import cmath as c
from numpy.linalg import norm

# UNDER 0.85 GeV

mpi=0.13957
mk=0.4957
s0=1.05**2
sh=1.45**2

err_par=5*0.02536488,5*0.03,4.78609247,0.02430731,0.03,0.0008,0.51826985,0.13930606,0.07246226,0.20114172,0.38666077,0.70489692
par=12.40636946,  10.06779414,  42.91144069,   0.31468154,         1.18099539,1.2754, -82.32091388, -42.66455895, -14.36629633,        10.16573076, -18.85504447,  14.16980074

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

em=2*mk

wm=w2(em**2)

def b01h(b1,b1h):
    return (b1*(s0/sh)*(np.sqrt(sh-em**2)/np.sqrt(s0-em**2))*((em+np.sqrt(sh-em**2))/(em+np.sqrt(s0-em**2)))**2 -2*b1h*wm)

def b0h(b0,b1,b1h):
    return b0 + b1*w(em**2)-b01h(b1,b1h)*wm-b1h*wm**2


def phi30(s,b0,b1,b1h,mf2):
    if s<em**2:
        return np.sqrt(s)*mpi**2*(mf2**2 - s)*(b0 + b1*w(s))/(2*q(s,mpi)**5)
    else:
        return np.sqrt(s)*mpi**2*(mf2**2 - s)*(b0h(b0,b1,b1h) + b01h(b1,b1h)*w2(s) + b1h*w2(s)**2)/(2*q(s,mpi)**5)

def deltaf1(s,b0,b1,b1h,eps,r,mf2):
    a=180*np.arctan(1/phi30(s,b0,b1,b1h,mf2)).real/np.pi
    if a>0:
        return a
    else:
        return a+180


def etaf1(s,b0,b1,b1h,eps,r,mf2):
    me=2*mk
    if s<me**2:
        return 1
    else:
        return 1-eps*((1-(me**2/s))/((1-(me**2/mf2**2))))**(5/2)*(1+r*(1-((s-me**2)/(mf2**2-me**2))))


def tf1(s,b0,b1,b1h,eps,r,mf2):
    return (etaf1(s,eps,r,mf2)*c.exp(np.pi*1j*deltaf1(s,b0,b1,b1h)/90)-1)/(2j*sigma(s,mpi))

"""Chebyshev continuation"""

sm=1.4**2

def w3(s):
    return (2*(np.sqrt(s)-1.4)/(2-1.4))-1

derW=(w3(sm+10**(-5))-w3(sm-10**(-5)))/(2*10**(-5))

def derD(b0,b1,b1h,eps,r,mf2):
    return (deltaf1(sm+10**(-5),b0,b1,b1h,eps,r,mf2)-deltaf1(sm-10**(-5),b0,b1,b1h,eps,r,mf2))/(2*10**(-5))

def derE(b0,b1,b1h,eps,r,mf2):
    return (etaf1(sm+10**(-5),b0,b1,b1h,eps,r,mf2)-etaf1(sm-10**(-5),b0,b1,b1h,eps,r,mf2))/(2*10**(-5))

def Delta(b0,b1,b1h,eps,r,mf2,d0,d1,d2):
    return (derD(b0,b1,b1h,eps,r,mf2)/derW) + 4*d0 - 9*d1 + 16*d2

def c1(s):
    return s

def c2(s):
    return 2*s**2 - 1

def c3(s):
    return 4*s**3 - 3*s

def c4(s):
    return 8*s**4 - 8*s**2 + 1

def deltaf2(s,b0,b1,b1h,eps,r,mf2,d0,d1,d2):
    return deltaf1(sm,b0,b1,b1h,eps,r,mf2) + Delta(b0,b1,b1h,eps,r,mf2,d0,d1,d2)*(c1(w3(s))+1) + d0*(c2(w3(s))-1) + d1*(c3(w3(s))+1) + d2*(c4(w3(s))-1)

def deltaf(s,b0,b1,b1h,eps,r,mf2,d0,d1,d2):
    if s<sm:
        return deltaf1(s,b0,b1,b1h,eps,r,mf2)
    else:
        return deltaf2(s,b0,b1,b1h,eps,r,mf2,d0,d1,d2)    

qm=q(sm,mpi)

def Q(s):
    return (s/sm)-1

def eps0(b0,b1,b1h,eps,r,mf2): 
    return c.sqrt(-c.log(etaf1(sm,b0,b1,b1h,eps,r,mf2)))

def eps1(b0,b1,b1h,eps,r,mf2): 
    return -sm*derE(b0,b1,b1h,eps,r,mf2)/(2*eps0(b0,b1,b1h,eps,r,mf2)*etaf1(sm,b0,b1,b1h,eps,r,mf2))                             

def etaf2(s,b0,b1,b1h,eps,r,mf2,eps2,eps3,eps4):
    return np.exp(-(eps0(b0,b1,b1h,eps,r,mf2) + eps1(b0,b1,b1h,eps,r,mf2)*Q(s) + eps2*Q(s)**2 + eps3*Q(s)**3 + eps4*Q(s)**4)**2)

def etaf(s,b0,b1,b1h,eps,r,mf2,d0,d1,d2,eps2,eps3,eps4):
    if s<sm:
        return etaf1(s,b0,b1,b1h,eps,r,mf2)
    else:
        return etaf2(s,b0,b1,b1h,eps,r,mf2,eps2,eps3,eps4).real

def tf(s,params):
    b0,b1,b1h,eps,r,mf2,d0,d1,d2,eps2,eps3,eps4=params
    return (etaf(s,b0,b1,b1h,eps,r,mf2,d0,d1,d2,eps2,eps3,eps4)*c.exp(np.pi*1j*deltaf(s,b0,b1,b1h,eps,r,mf2,d0,d1,d2)/90)-1)/(2j*sigma(s,mpi))

def deltafr(s,params):
    b0,b1,b1h,eps,r,mf2,d0,d1,d2,eps2,eps3,eps4=params
    return deltaf(s,b0,b1,b1h,eps,r,mf2,d0,d1,d2)

def etafr(s,params):
    b0,b1,b1h,eps,r,mf2,d0,d1,d2,eps2,eps3,eps4=params
    return etaf(s,b0,b1,b1h,eps,r,mf2,d0,d1,d2,eps2,eps3,eps4)

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
    return erru(deltafr,s,parf,err_parf)
def errdd(s,parf,err_parf):
    return errd(deltafr,s,parf,err_parf)

def loge(s,params):
    return np.sqrt(-np.log(etafr(s,params)))

def erreu(s,parf,err_parf):
    a=etafr(s,parf)
    b=np.exp(-(erru(loge,s,parf,err_parf)+loge(s,parf))**2)
    c=np.exp(-(errd(loge,s,parf,err_parf)+loge(s,parf))**2)
    if a<b or a<c:
        return max(b,c)
    else:
        return a

def erred(s,parf,err_parf):
    a=etafr(s,parf)
    b=np.exp(-(erru(loge,s,parf,err_parf)+loge(s,parf))**2)
    c=np.exp(-(errd(loge,s,parf,err_parf)+loge(s,parf))**2)
    if a>b or a>c:
        return min(b,c)
    else:
        return a
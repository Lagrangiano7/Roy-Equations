"""New P-wave parameterization (piecewise) for Solution II"""

import numpy as np
import cmath as c
from numpy.linalg import norm


mpi=0.13957
mk=0.4957
s0=1.43**2
alpha=0.3
mpi0=mpi
mom=0.78266

N1=17

par=1.16780592, -1.18591532,  1.51692843,  3.44055905, -2.58271532,        0.48501437, -0.71111251,  0.3358241 , -0.05135191,  0.77045951,        1.7504339 ,  0.86252597, -0.24647558, -0.01831969,  0.07939658
err_par=N1*0.00376019,0.0162618,0.047176,0.0943307,0.10797209,0.00147,0.000827,0.000488,0.000307,0.0012,0.3754746,0.10167225,0.01148617,0.00305172, 0.00156105


def sigma(s,m):
    if s.imag>=0:
        return c.sqrt(1-(4*m**2/s))
    else:
        return -c.sqrt(1-(4*m**2/s))

def sigma2(s,m):
    return c.sqrt(1-(4*m**2/s))
    
def q(s,m):
    return c.sqrt((s/4)-m**2)

def v(s):
    return (c.sqrt(s)-alpha*c.sqrt(s0-s))/(c.sqrt(s)+alpha*c.sqrt(s0-s))

def w(s):
    return v(s+1j*10**(-18))

def phi11(s,b0,b1,b2,b3,b4,mrho):
    tt=c.sqrt(s)*(mrho**2 - s)*(((2*mpi**3)/(mrho**2*c.sqrt(s)))+b0 + b1*w(s)+ b2*w(s)**2+ b3*w(s)**3+ b4*w(s)**4)/(2*q(s,mpi)**3)
    if s.imag>=0:
        return tt
    else:
        return-tt

def tc2(s,b0,b1,b2,b3,b4,mrho):
    return 1/(sigma(s,mpi)*(phi11(s,b0,b1,b2,b3,b4,mrho)-1j))

def Sc2(s,b0,b1,b2,b3,b4,mrho):
    return 1+ 2j*sigma(s,mpi)*tc2(s,b0,b1,b2,b3,b4,mrho)

def deltac2(s,b0,b1,b2,b3,b4,mrho):
    a= 90*c.phase(Sc2(s,b0,b1,b2,b3,b4,mrho))/np.pi
    if a>0:
        return a
    else:
        return a+180
    
# OVER 0.917 GeV

sm1=(mpi0+mom)**2

def nu(s,m1,m2):
    return c.sqrt((s-(m1+m2)**2)*(s-(m1-m2)**2))

def J(s,m1,m2):
    Delta=m1**2-m2**2
    Sigma=m1**2+m2**2
    return (2+(Delta/s -Sigma/Delta)*np.log(m2**2/m1**2)+(nu(s,m1,m2)*(c.log((nu(s,m1,m2)-s+Delta)/(nu(s,m1,m2)+s+Delta))+c.log((nu(s,m1,m2)-s-Delta)/(nu(s,m1,m2)+s-Delta)))/s))/(2*np.pi)

mp=(mpi0+mom)

def R(s):
    return s/mp**2 -1

def deltain(s,k0,k1,k2,k3):
    q2=(s-(mpi0+mom)**2)*(s-(mpi0-mom)**2)/(4*s)
    return  J(s,mpi0,mom)*(k0+k1*R(s)+k2*R(s)**2+k3*R(s)**3)*q(s,mpi)**3*q2/(c.sqrt(s)*mpi**2)/(mp**2)


def tin2(s,k0,k1,k2,k3):
    return  (np.exp(2j*deltain(s,k0,k1,k2,k3))-1)/(2j*sigma(s,mpi))

def Sin2(s,k0,k1,k2,k3):
    return 1+ 2j*sigma(s,mpi)*tin2(s,k0,k1,k2,k3)


def S1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return Sc2(s,b0,b1,b2,b3,b4,mrho)*Sin2(s,k0,k1,k2,k3)

def etaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return abs(S1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho))


def deltaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    a= 90*c.phase(S1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho))/np.pi
    if a>0:
        return a
    else:
        return a+180

def tf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return  (etaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)*np.exp(1j*deltaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)*90/np.pi)-1)/(2j*sigma(s,mpi))

# OVER 1.4 GeV

sm=1.4**2

def c1(s):
    return s

def c2(s):
    return 2*s**2 - 1

def c3(s):
    return 4*s**3 - 3*s

def c4(s):
    return 8*s**4 - 8*s**2 + 1

def c5(s):
    return 16*s**5 - 20*s**3 + 5*s

def c6(s):
    return 32*s**6 - 48*s**4 + 18*s**2-1

def c7(s):
    return 64*s**7 - 112*s**5 + 56*s**3 - 7*s

def w2(s):
    return (2*(np.sqrt(s)-1.4)/(2-1.4))-1

derW=(w2(sm+10**(-5))-w2(sm-10**(-5)))/(2*10**(-5))

def derD(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return (deltaf1(sm+10**(-5),b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)-deltaf1(sm-10**(-5),b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho))/(2*10**(-5))

def derE(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return (etaf1(sm+10**(-5),b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)-etaf1(sm-10**(-5),b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho))/(2*10**(-5))

def Delta(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d0,d1):
    return (derD(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)/derW) + 4*d0 - 9*d1 

def deltaf2(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d0,d1):
    return deltaf1(sm,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho) + Delta(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d0,d1)*(c1(w2(s))+1) + d0*(c2(w2(s))-1) + d1*(c3(w2(s))+1) 

def deltaf(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d0,d1):
    if s.real<sm:
        return deltaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)
    else:
        return deltaf2(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d0,d1)

def eps0(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho):
    return np.sqrt(-np.log(etaf1(sm,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)))

def eps1(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4):
    return -derE(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)/(2*eps0(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)*etaf1(sm,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)*derW) +4*eps2 -9*eps3 +16*eps4                             
    
def eta2(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4):
    return np.exp(-(eps0(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho) + eps1(b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4)*(c1(w2(s))+1) + eps2*(c2(w2(s))-1) + eps3*(c3(w2(s))+1) + eps4*(c4(w2(s))-1))**2)

def etaf(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4):
    if s.real<sm:
        return etaf1(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho)
    else:
        return eta2(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4).real

def tf(s,params):
    b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d0,d1,eps2,eps3,eps4=params 
    return (etaf(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4)*c.exp(np.pi*1j*deltaf(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d0,d1)/90)-1)/(2j*sigma(s,mpi))

def deltafr(s,params):
    b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d0,d1,eps2,eps3,eps4=params
    return deltaf(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d0,d1)

def etafr(s,params):
    b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,d0,d1,eps2,eps3,eps4=params
    return etaf(s,b0,b1,b2,b3,b4,k0,k1,k2,k3,mrho,eps2,eps3,eps4)

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
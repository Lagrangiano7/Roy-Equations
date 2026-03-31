"""S0 wave Solution III"""

import numpy as np
import cmath as c
from numpy.linalg import norm

# UNDER 1.4 GeV

# First, the conformal amplitude

mpi=0.13957
mk=0.4957
s0=4*mk**2
sm=1.4**2
z0=0.136
err_z0=0.035

N=1

# b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2,eps2,eps3,eps4
par=12.3,-1.0,15.7,-6.0,-22.1,7.1,0.996,0.025,5.26,-4.64,0.10,-0.29,73.4,27.3,-0.3,171.6,-1038.8,1704.7
err_par=0.3,0.9,1.7,1.6,1.2,2.8,0,0,0.08,0.04,0.07,0.04,1.5,0.4,0.2,2,8.3,30.8


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

def phi00(s,b0,b1,b2,b3,b4,b5):
    return c.sqrt(s)*mpi**2*((z0**2/(mpi*c.sqrt(s)))+b0 + b1*w(s)+ b2*w(s)**2+ b3*w(s)**3+ b4*w(s)**4+ b5*w(s)**5)/(2*q(s,mpi)*(s-(z0**2/2)))

def tc2(s,b0,b1,b2,b3,b4,b5):
    return 1/(sigma(s,mpi)*(phi00(s,b0,b1,b2,b3,b4,b5)-1j))

# f0(980) amplitude

def J(s,m):
    return (2+sigma(s,m)*c.log((sigma(s,m)-1)/(sigma(s,m)+1)))/np.pi

def w1(s):
    return (2*(c.sqrt(s)-2*mk)/(1.5-2*mk))-1

def c1(s):
    return s

def c2(s):
    return 2*s**2 - 1

def c3(s):
    return 4*s**3 - 3*s

def f(s,k0,k1,k2,k3):
    return mk**2*(k0+k1*c1(w1(s))+k2*c2(w1(s))+k3*c3(w1(s)))

def sp(rsp,isp):
    return (rsp - isp*1j)**2

def fr(rsp,isp,k0,k1,k2,k3):
    return f(sp(rsp,isp),k0,k1,k2,k3).real

def fi(rsp,isp,k0,k1,k2,k3):
    return f(sp(rsp,isp),k0,k1,k2,k3).imag

def Jkr(rsp,isp):
    return J(sp(rsp,isp),mk).real

def Jki(rsp,isp):
    return J(sp(rsp,isp),mk).imag

def Jpr(rsp,isp):
    return J(sp(rsp,isp),mpi).real

def Jpi(rsp,isp):
    return J(sp(rsp,isp),mpi).imag

def sigmar(rsp,isp):
    return sigma(sp(rsp,isp),mpi).real

def sigmai(rsp,isp):
    return sigma(sp(rsp,isp),mpi).imag

def sr(rsp,isp):
    return sp(rsp,isp).real

def si(rsp,isp):
    return sp(rsp,isp).imag

def d(rsp,isp):
    return Jpi(rsp,isp)*sr(rsp,isp) + Jpr(rsp,isp)*si(rsp,isp) + 2*(si(rsp,isp)*sigmai(rsp,isp) - sr(rsp,isp)*sigmar(rsp,isp))

def G(rsp,isp,k0,k1,k2,k3):
    return -((fi(rsp,isp,k0,k1,k2,k3)*Jkr(rsp,isp) + fr(rsp,isp,k0,k1,k2,k3)*Jki(rsp,isp) + si(rsp,isp))/d(rsp,isp))

def M(rsp,isp,k0,k1,k2,k3):
    return ((((fi(rsp,isp,k0,k1,k2,k3)*Jkr(rsp,isp) + fr(rsp,isp,k0,k1,k2,k3)*Jki(rsp,isp))*(si(rsp,isp)*(Jpi(rsp,isp) - 2*sigmar(rsp,isp)) - sr(rsp,isp)*(Jpr(rsp,isp) + 2*sigmai(rsp,isp)))) + (Jpi(rsp,isp) - 2*sigmar(rsp,isp))*(si(rsp,isp)**2 + sr(rsp,isp)**2))/d(rsp,isp)) - (fi(rsp,isp,k0,k1,k2,k3)*Jki(rsp,isp) - fr(rsp,isp,k0,k1,k2,k3)*Jkr(rsp,isp))

def tf2(s,rsp,isp,k0,k1,k2,k3):
    return s*G(rsp,isp,k0,k1,k2,k3)/(M(rsp,isp,k0,k1,k2,k3)-s-J(s,mpi)*s*G(rsp,isp,k0,k1,k2,k3)-J(s,mk)*f(s,k0,k1,k2,k3))

# Let's define the total amplitude

def t12(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3):
    return tc2(s,b0,b1,b2,b3,b4,b5) + tf2(s,rsp,isp,k0,k1,k2,k3) + 2j*sigma(s,mpi)*tc2(s,b0,b1,b2,b3,b4,b5)*tf2(s,rsp,isp,k0,k1,k2,k3)

def S1(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3):
    return 1+ 2j*sigma(s,mpi)*t12(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3)

def eta11(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3):
    return abs(S1(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3))

def delta11(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3):
    a=90*c.phase(S1(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3))/np.pi
    if s>1.1**2 and a<0:
        return  a+360
    elif a<0 or s>0.98**2:
        return a+180
    else:
        return a

# OVER 1.4 GeV

def w2(s):
    return (2*(np.sqrt(s)-1.4)/(2-1.4))-1

derW=(w2(sm+10**(-5))-w2(sm-10**(-5)))/(2*10**(-5))

def derD(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3):
    return (delta11(sm+10**(-5),b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3)-delta11(sm-10**(-5),b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3))/(2*10**(-5))

def derE(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3):
    return (eta11(sm+10**(-5),b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3)-eta11(sm-10**(-5),b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3))/(2*10**(-5))

def Delta(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2):
    return (derD(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3)/derW) + 4*d0 - 9*d1 + 16*d2

def c4(s):
    return 8*s**4 - 8*s**2 + 1

def delta2(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2):
    return delta11(sm,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3) + Delta(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2)*(c1(w2(s))+1) + d0*(c2(w2(s))-1) + d1*(c3(w2(s))+1) + d2*(c4(w2(s))-1)

def deltaf(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2):
    if s<sm:
        return delta11(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3)
    else:
        return delta2(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2)    

qm=q(sm,mpi)

def Q(s):
    return (q(s,mpi)/qm)-1

def eps0(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3): 
    return c.sqrt(-c.log(eta11(sm,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3)))

def eps1(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3): 
    return -4*qm**2*derE(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3)/(eps0(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3)*eta11(sm,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3))                              

def eta2(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,eps2,eps3,eps4):
    return np.exp(-(eps0(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3) + eps1(b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3)*Q(s) + eps2*Q(s)**2 + eps3*Q(s)**3 + eps4*Q(s)**4)**2)

def etaf(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,eps2,eps3,eps4):
    if s<sm:
        return eta11(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3)
    else:
        return eta2(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,eps2,eps3,eps4).real

def tf(s,params):
    b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2,eps2,eps3,eps4=params
    return (etaf(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,eps2,eps3,eps4)*c.exp(np.pi*1j*deltaf(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2)/90)-1)/(2j*sigma(s,mpi))

def deltaf2(s,params):
    b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2,eps2,eps3,eps4=params
    return deltaf(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2)

def etaf2(s,params):
    b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,d0,d1,d2,eps2,eps3,eps4=params
    return etaf(s,b0,b1,b2,b3,b4,b5,rsp,isp,k0,k1,k2,k3,eps2,eps3,eps4)

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

def erru2(f,s,parf,err_parf):
    n=len(parf)
    r=np.zeros(n)
    fv=f(s,parf)
    for i in range(0,n):
        par1=list(parf)
        par1[i]=par1[i]+err_parf[i]
        w1=f(s,par1)-fv
        r[i]=w1
    return norm(r)

def errd2(f,s,parf,err_parf):
    n=len(parf)
    r=np.zeros(n)
    fv=f(s,parf)
    for i in range(0,n):
        par1=list(parf)
        par1[i]=par1[i]+err_parf[i]
        w1=f(s,par1)-fv
        r[i]=w1
    return - norm(r)

def errdu(s,parf,err_parf):
    return erru(deltaf2,s,parf,err_parf)
def errdd(s,parf,err_parf):
    return errd(deltaf2,s,parf,err_parf)

def loge(s,params):
    return np.sqrt(-np.log(etaf2(s,params)))

def erreu(s,parf,err_parf):
    return np.exp(-(erru(loge,s,parf,err_parf)+loge(s,parf))**2)
def erred(s,parf,err_parf):
    return np.exp(-(errd(loge,s,parf,err_parf)+loge(s,parf))**2)
import numpy as np
import scipy.integrate as integrate
from scipy.special import eval_legendre
from .gaussxw import gaussxwab

############ Global ############
mpi = 0.13957
sth = (2*mpi)**2 # Threshold
s2 = 1.42**2 # Matching point
arhop = 0.90
arhopp = -0.3
apPp = 0.90
appPp = -0.3 # DOES THIS HAVE NONZERO ERROR????????????????????????????????????????????????

eps=1e-6
N_gauss = 10
x, w_x = gaussxwab(N_gauss, -1, 1)

############ Parameters and errors ############

params_Regge = {
    "betarho": 1.48224009,
    "b": 2.4,
    "arho0": 0.443163931,
    "drho": 1.88883736,
    "erho": 2.67938016,
    "CP": 0.618790418,
    "CPp": -0.378777533,
    "apP": 0.2,
    "betaP": 2.4978,
    "aPp0": 0.53,
    "betaPp": 0.796,
    "beta2": 0.0779179905
}

errs_Regge = {
    "betarho": 0.14,
    "b": 0.2,
    "arho0": 0.02,
    "drho": 0.5,
    "erho": 2.5,
    "CP": 1.0,
    "CPp": 0.4,
    "apP": 0.1,
    "betaP": 0.04,
    "aPp0": 0.02,
    "betaPp": 0.05,
    "beta2": 0.2
}

############ ImT1 ############

def ImT1t(s, t, params):
    arho = lambda t: params["arho0"] + arhop*t + 1/2*arhopp*t**2
    varphi = lambda t: 1 + params["drho"]*t + params["erho"]*t**2
    return params["betarho"]*(1 + arho(t))/(1 + params["arho0"])*varphi(t)*np.exp(params["b"]*t)*s**arho(t)


############ ImT0 ############


def ImT0t(s, t, params):
    aP = lambda t: 1 + params["apP"]*t
    PsiP = lambda t: 1 + params["CP"]*t
    P = lambda s, t: params["betaP"]*PsiP(t)*aP(t)*(1 + aP(t))/2*np.exp(params["b"]*t)*s**aP(t)

    aPp = lambda t: params["aPp0"] + apPp*t + appPp*t**2
    PsiPp = lambda t: 1 + params["CPp"]*t
    Pp = lambda s, t: params["betaPp"]*PsiPp(t)*aPp(t)*(1 + aPp(t))/(params["aPp0"]*(1 + params["aPp0"]))*np.exp(params["b"]*t)*s**aPp(t)

    return P(s, t) + Pp(s, t)

############ ImT2 ############

def ImT2t(s, t, params):
    arho = lambda t: params["arho0"] + arhop*t + 1/2*arhopp*t**2
    return params["beta2"]*np.exp(params["b"]*t)*s**(arho(t) + params["arho0"] - 1)

############ Re T in s-channel ############

II = np.eye(3)
Cst = np.array([[1/3, 1, 5/3], [1/3, 1/2, -5/6], [1/3, -1/2, 1/6]])
Ctu = np.array([[1,0,0], [0,-1,0], [0,0,1]])
Csu = np.array([[1/3, -1, 5/3], [-1/3, 1/2, 5/6], [1/3, 1/2, 1/6]])

tt = lambda s, z: 1/2*(sth-s)*(1-z)

g2 = lambda s, t, sp: -t/(np.pi*sp*(sp-sth)) * np.matmul(( (sth-s-t)*Cst + s*np.matmul(Cst, Ctu) ), II/(sp-t) + Csu/(sp-sth+t))
g3 = lambda s, t, sp: -s*(sth-s-t)/(np.pi*sp*(sp-sth+t))*(II/(sp-s) + Csu/(sp-sth+s+t))

ImTI = lambda s, t, params: 4*np.pi**2 * np.matmul(Cst, np.array([[ImT0t(s, t, params)], [ImT1t(s, t, params)], [ImT2t(s, t, params)]]))
integrand = lambda s, t, sp, params: np.matmul(g2(s, t, sp), ImTI(sp, 0, params)) + np.matmul(g3(s, t, sp), ImTI(sp, t, params))

integrandI0 = lambda s, t, sp, params: integrand(s, t, sp, params)[0]
integrandI1 = lambda s, t, sp, params: integrand(s, t, sp, params)[1]
integrandI2 = lambda s, t, sp, params: integrand(s, t, sp, params)[2]

def ReT0(s, t, params):
    sp = lambda z: z/(1-z)+s2
    return integrate.quad(lambda z: 1/(1-z)**2 * integrandI0(s, t, sp(z), params), 0, 1-1e-6)[0]

def ReT1(s, t, params):
    sp = lambda z: z/(1-z)+s2
    return integrate.quad(lambda z: 1/(1-z)**2 * integrandI1(s, t, sp(z), params), 0, 1-1e-6)[0]

def ReT2(s, t, params):
    sp = lambda z: z/(1-z)+s2
    return integrate.quad(lambda z: 1/(1-z)**2 * integrandI2(s, t, sp(z), params), 0, 1-1e-6)[0]

Ret00 = lambda s, params=params_Regge: 1/(64*np.pi)*np.sum(w_x * np.array([ReT0(s, tt(s, xi), params) for xi in x]))

Ret02 = lambda s, params=params_Regge: 1/(64*np.pi)*np.sum(w_x * eval_legendre(2,x)*np.array([ReT0(s, tt(s, xi), params) for xi in x]))

Ret04 = lambda s, params=params_Regge: 1/(64*np.pi)*np.sum(w_x * eval_legendre(4,x)*np.array([ReT0(s, tt(s, xi), params) for xi in x]))

Ret11 = lambda s, params=params_Regge: 1/(64*np.pi)*np.sum(w_x * eval_legendre(1, x)* np.array([ReT1(s, tt(s, xi), params) for xi in x]))

Ret13 = lambda s, params=params_Regge: 1/(64*np.pi)*np.sum(w_x * eval_legendre(3,x)*np.array([ReT1(s, tt(s, xi), params) for xi in x]))

Ret20 = lambda s, params=params_Regge: 1/(64*np.pi)*np.sum(w_x * np.array([ReT2(s, tt(s, xi), params) for xi in x]))

Ret22 = lambda s, params=params_Regge: 1/(64*np.pi)*np.sum(w_x * eval_legendre(2,x)*np.array([ReT2(s, tt(s, xi), params) for xi in x]))

Ret24 = lambda s, params=params_Regge: 1/(64*np.pi)*np.sum(w_x * eval_legendre(4,x)*np.array([ReT2(s, tt(s, xi), params) for xi in x]))
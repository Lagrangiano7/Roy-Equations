import numpy as np
import scipy.integrate as integrate
from scipy.special import eval_legendre

############ Global ############
mpi = 0.13957
sth = (2*mpi)**2 # Threshold
s2 = 1.42**2 # Matching point

############ ImT1 ############

betarho = 1.48224009
b = 2.4
arho0 = 0.443163931
arhop = 0.90
arhopp = -0.3
drho = 1.88883736
erho = 2.67938016
arho = lambda t: arho0 + arhop*t + 1/2*arhopp*t**2
varphi = lambda t: 1 + drho*t + erho*t**2
ImT1t = lambda s, t: betarho*(1 + arho(t))/(1 + arho0)*varphi(t)*np.exp(b*t)*s**arho(t)

############ ImT0 ############

CP = 0.618790418
CPp = -0.378777533
apP = 0.2
aP = lambda t: 1 + apP*t
PsiP = lambda t: 1 + CP*t
betaP = 2.4978

P = lambda s, t: betaP*PsiP(t)*aP(t)*(1 + aP(t))/2*np.exp(b*t)*s**aP(t)

aPp0 = 0.53
apPp = 0.90
appPp = -0.3
aPp = lambda t: aPp0 + apPp*t + apPp*t**2
CPp = -0.378
PsiPp = lambda t: 1 + CPp*t
betaPp = 0.796

Pp = lambda s, t: betaPp*PsiPp(t)*aPp(t)*(1 + aPp(t))/(aPp0*(1 + aPp0))*np.exp(b*t)*s**aPp(t)
ImT0t = lambda s, t: P(s, t) + Pp(s, t)

############ ImT2 ############

beta2 = 0.0779179905
ImT2t = lambda s, t: beta2*np.exp(b*t)*s**(arho(t) + arho0 - 1)

############ Re T in s-channel ############

II = np.eye(3)
Cst = np.array([[1/3, 1, 5/3], [1/3, 1/2, -5/6], [1/3, -1/2, 1/6]])
Ctu = np.array([[1,0,0], [0,-1,0], [0,0,1]])
Csu = np.array([[1/3, -1, 5/3], [-1/3, 1/2, 5/6], [1/3, 1/2, 1/6]])

tt = lambda s, z: 1/2*(sth-s)*(1-z)

g2 = lambda s, t, sp: -t/(np.pi*sp*(sp-sth)) * np.matmul(( (sth-s-t)*Cst + s*np.matmul(Cst, Ctu) ), II/(sp-t) + Csu/(sp-sth+t))
g3 = lambda s, t, sp: -s*(sth-s-t)/(np.pi*sp*(sp-sth+t))*(II/(sp-s) + Csu/(sp-sth+s+t))

ImTI = lambda s, t: 4*np.pi**2 * np.matmul(Cst, np.array([[ImT0t(s, t)], [ImT1t(s, t)], [ImT2t(s, t)]]))
integrand = lambda s, t, sp: np.matmul(g2(s, t, sp), ImTI(sp, 0)) + np.matmul(g3(s, t, sp), ImTI(sp, t))

integrandI0 = lambda s, t, sp: integrand(s, t, sp)[0]
integrandI1 = lambda s, t, sp: integrand(s, t, sp)[1]
integrandI2 = lambda s, t, sp: integrand(s, t, sp)[2]

def ReT0(s, t):
    sp = lambda z: z/(1-z)+s2
    return integrate.quad(lambda z: 1/(1-z)**2 * integrandI0(s, t, sp(z)), 0, 1-1e-6)[0]

def ReT1(s, t):
    sp = lambda z: z/(1-z)+s2
    return integrate.quad(lambda z: 1/(1-z)**2 * integrandI1(s, t, sp(z)), 0, 1-1e-6)[0]

def ReT2(s, t):
    sp = lambda z: z/(1-z)+s2
    return integrate.quad(lambda z: 1/(1-z)**2 * integrandI2(s, t, sp(z)), 0, 1-1e-6)[0]

Ret00 = lambda s: 1/(64*np.pi)*integrate.quad(lambda x: ReT0(s, tt(s, x)), -1, 1)[0]

Ret11 = lambda s: 1/(64*np.pi)*integrate.quad(lambda x: x*ReT1(s, tt(s, x)), -1, 1)[0]

Ret20 = lambda s: 1/(64*np.pi)*integrate.quad(lambda x: ReT2(s, tt(s, x)), -1, 1)[0]

Ret02 = lambda s: 1/(64*np.pi)*integrate.quad(lambda x: eval_legendre(2, x)*ReT0(s, tt(s, x)), -1, 1)[0]

Ret22 = lambda s: 1/(64*np.pi)*integrate.quad(lambda x: eval_legendre(2, x)*ReT2(s, tt(s, x)), -1, 1)[0]

Ret13 = lambda s: 1/(64*np.pi)*integrate.quad(lambda x: eval_legendre(3, x)*ReT1(s, tt(s, x)), -1, 1)[0]

Ret04 = lambda s: 1/(64*np.pi)*integrate.quad(lambda x: eval_legendre(4, x)*ReT0(s, tt(s, x)), -1, 1)[0]

Ret24 = lambda s: 1/(64*np.pi)*integrate.quad(lambda x: eval_legendre(4, x)*ReT2(s, tt(s, x)), -1, 1)[0]
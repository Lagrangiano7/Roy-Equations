import numpy as np
import scipy.integrate as integrate
import kernels
from kernels import sth, s2
import t
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from Regge import reader
from gaussxw import gaussxwab

eps=1e-6
wave_ord = t.waves_1[1:]
K_ord = kernels.getS0()
STS0 = lambda s: kernels.ST_S0(s, kernels.a00_1, kernels.a20_1)

# Interpolating parametrizations for shorter integration time

sp_grid = np.linspace(sth+eps, s2, 1500)

# Diagonal (t00)
Im_t00_vals = np.array([t.t00_1(sp).imag for sp in sp_grid])
Im_t00_interp = PchipInterpolator(sp_grid, Im_t00_vals)

# Ondas no diagonales
Im_interp = []
for i in range(len(K_ord)):
    vals = np.array([wave_ord[i](sp).imag for sp in sp_grid])
    Im_interp.append(PchipInterpolator(sp_grid, vals))

# =========================================================

N=50
x, w = gaussxwab(N, sth+eps,s2)

def ReT00(s):
    # S0 -> S0
    cauchy_kern = lambda sp: kernels.K0000_cauchy(s, sp) * Im_t00_interp(sp)
    cauchy_rest = lambda sp: kernels.K0000_rest(s, sp) * Im_t00_interp(sp)

    total_diag = integrate.quad(
        cauchy_kern,
        sth+eps, s2,
        weight="cauchy",
        wvar=s,
        limit=200
    )[0]
    for j in range(N):
        total_diag+=w[j]*cauchy_rest(x[j])

    # Rest -> S0
    total_rest = 0
    for i in range(len(K_ord)):
        integrand = lambda sp: K_ord[i](s, sp) * Im_interp[i](sp)
        for j in range(N):
            total_rest+=w[j]*integrand(x[j])
    
    return STS0(s) + total_diag + total_rest


# =========================================================

x1 = np.linspace(np.sqrt(sth+1e-3), np.sqrt(68)*kernels.mpi, 600)

y_teoLo = np.array(list(map(lambda s: ReT00(s**2), x1)))
y_ReggeHI = np.array(list(map(lambda s: reader.getWave("S0")(s**2), x1)))

y_teo = y_teoLo + y_ReggeHI
y_param = list(map(lambda s: t.t00_1(s**2).real, x1))

raw = np.loadtxt("roys0.out")

x = raw[:,0]/1000
Ret00_Jacobo_disp = raw[:,1]
Ret00_Jacobo_param = raw[:,2]

plt.plot(x, Ret00_Jacobo_disp, label="Dispersivo Jacobo")
plt.plot(x, Ret00_Jacobo_param, label="Param Jacobo")

plt.plot(x1, y_teo, label="Roy")
plt.plot(x1, y_param, label="Parametrización")
plt.xlabel("$\\sqrt{s}$ (GeV)")
plt.ylabel("Re t00(s)")
plt.legend()
plt.show()

import numpy as np
import scipy.integrate as integrate
import kernels
from kernels import sth, s2
import t
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from Regge import reader

eps=1e-6
wave_ord = np.delete(t.waves_1, 6)
K_ord = kernels.getD2()

# Interpolating parametrizations for shorter integration time

sp_grid = np.linspace(sth+eps, s2, 1500)

# Diagonal (t00)
Im_t22_vals = np.array([t.t22_1(sp).imag for sp in sp_grid])
Im_t22_interp = PchipInterpolator(sp_grid, Im_t22_vals)

# Ondas no diagonales
Im_interp = []
for i in range(len(K_ord)):
    vals = np.array([wave_ord[i](sp).imag for sp in sp_grid])
    Im_interp.append(PchipInterpolator(sp_grid, vals))

# =========================================================


def ReT22(s):
    # S0 -> S0
    cauchy_kern = lambda sp: kernels.K2222_cauchy(s, sp) * Im_t22_interp(sp)
    cauchy_rest = lambda sp: kernels.K2222_rest(s, sp) * Im_t22_interp(sp)

    total_diag = integrate.quad(
        cauchy_kern,
        sth+eps, s2,
        weight="cauchy",
        wvar=s,
        limit=500
    )[0] + integrate.quad(
        cauchy_rest,
        sth+eps, s2,
        limit=500
    )[0]

    # Rest -> S0
    total_rest = 0
    for i in range(len(K_ord)):
        integrand = lambda sp: K_ord[i](s, sp) * Im_interp[i](sp)
        total_rest += integrate.quad(
            integrand,
            sth+eps, s2,
            limit=500
            )[0]
    
    return total_diag + total_rest


# =========================================================

x1 = np.linspace(np.sqrt(sth+1e-1), np.sqrt(68)*kernels.mpi, 800)

y_teoLo = np.array(list(map(lambda s: ReT22(s**2), x1)))
y_ReggeHI = np.array(list(map(lambda s: reader.getWave("D2")(s**2), x1)))

y_teo = y_teoLo + y_ReggeHI
y_param = list(map(lambda s: t.t22_1(s**2).real, x1))

plt.plot(x1, y_teo, label="Roy")
plt.plot(x1, y_param, label="Parametrización")
plt.xlabel("$\\sqrt{s}$ (GeV)")
plt.ylabel("Re t00(s)")
plt.legend()
plt.show()

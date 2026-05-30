import numpy as np
import scipy.integrate as integrate
import kernels
from kernels import sth, s2
import t
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from Regge import Regge

eps=1e-6

KT_waves = [t.t00_1, t.t20_1] # S0, S2
DT_waves = [t.t02_1, t.t04_1, t.t13_1, t.t22_1, t.t24_1] # D0, G0, F, D2, G2
K_ord = kernels.getP1()
KT = [K_ord[0], K_ord[4]] # S0, S2
DT = [K_ord[1], K_ord[2], K_ord[3], K_ord[5], K_ord[6]] # D0, G0, F, D2, G2
STP1 = lambda s: kernels.ST_P1(s, kernels.a00_1, kernels.a20_1)

# Interpolating parametrizations for shorter integration time

sp_grid = np.linspace(sth+eps, s2, 400)

# Diagonal (t11)
Im_t11_vals = np.array([t.t11_1(sp).imag for sp in sp_grid])
Im_t11_interp = PchipInterpolator(sp_grid, Im_t11_vals)

# Ondas no diagonales
Im_interp_KT = []
Im_interp_DT = []
for wave in KT_waves:
    vals = np.array([wave(sp).imag for sp in sp_grid])
    Im_interp_KT.append(PchipInterpolator(sp_grid, vals))

for wave in DT_waves:
    vals = np.array([wave(sp).imag for sp in sp_grid])
    Im_interp_DT.append(PchipInterpolator(sp_grid, vals))

# =========================================================


def ReT11(s):
    # KT (S0, P, S2)
    cauchy_kern = lambda sp: kernels.K1111_cauchy(s, sp) * Im_t11_interp(sp)
    cauchy_rest = lambda sp: kernels.K1111_rest(s, sp) * Im_t11_interp(sp)

    total_diag = integrate.quad(
        cauchy_kern,
        sth+eps, s2,
        weight="cauchy",
        wvar=s,
        limit=200
    )[0] + integrate.quad(
        cauchy_rest,
        sth+eps, s2,
        limit=200
    )[0]

    KT_contrib = total_diag
    for i in range(len(KT)):
        integrand = lambda sp: KT[i](s, sp) * Im_interp_KT[i](sp)
        KT_contrib += integrate.quad(
            integrand,
            sth+eps, s2,
            limit=200
            )[0]

    # DT
    DT_contrib = 0
    for i in range(len(DT)):
        integrand = lambda sp: DT[i](s, sp) * Im_interp_DT[i](sp)
        DT_contrib += integrate.quad(
            integrand,
            sth+eps, s2,
            limit=200
            )[0]
    
    return (STP1(s)+ KT_contrib+ DT_contrib)


# =========================================================

x1 = np.linspace(np.sqrt(sth+1e-3), np.sqrt(68)*kernels.mpi, 50)

y_teoLo100 = np.array(list(map(lambda s: ReT11(s**2), x1)))
y_ReggeHI = np.array(list(map(lambda s: Regge.Ret11(s**2), x1)))

y_teo100 = y_teoLo100 + y_ReggeHI
y_param = list(map(lambda s: t.t11_1(s**2).real, x1))



plt.plot(x1, y_teo100, label="Roy")
plt.plot(x1, y_param, label="Parametrization")
plt.xlabel("$\\sqrt{s}$ (GeV)")
plt.ylabel("Re t11(s)")
plt.legend()
plt.show()
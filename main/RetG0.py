import numpy as np
import scipy.integrate as integrate
import kernels
from kernels import sth, s2
import t
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from Regge import Regge

eps=1e-6

KT_waves = [t.t00_1, t.t11_1, t.t20_1] # S0, P, S2
DT_waves = [t.t02_1, t.t13_1, t.t22_1, t.t24_1] # D0, F, D2, G2

KT = [kernels.K0400, kernels.K0411, kernels.K0420] # S0, P, S2
DT = [kernels.K0402, kernels.K0413, kernels.K0422, kernels.K0424] # D0, F, D2, G2

# Interpolating parametrizations for shorter integration time

sp_grid = np.linspace(sth+eps, s2, 800)

# Diagonal (t04)
Im_t04_vals = np.array([t.t04_1(sp).imag for sp in sp_grid])
Im_t04_interp = PchipInterpolator(sp_grid, Im_t04_vals)

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


def ReT04(s):
    # KT (S0, P, S2)

    KT_contrib = 0
    for i in range(len(KT)):
        integrand = lambda sp: KT[i](s, sp) * Im_interp_KT[i](sp)
        KT_contrib += integrate.quad(
            integrand,
            sth+eps, s2,
            limit=500
            )[0]

    # DT
    cauchy_kern = lambda sp: kernels.K0404_cauchy(s, sp) * Im_t04_interp(sp)
    cauchy_rest = lambda sp: kernels.K0404_rest(s, sp) * Im_t04_interp(sp)

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

    DT_contrib = total_diag
    for i in range(len(DT)):
        integrand = lambda sp: DT[i](s, sp) * Im_interp_DT[i](sp)
        DT_contrib += integrate.quad(
            integrand,
            sth+eps, s2,
            limit=500
            )[0]
    
    return (KT_contrib, DT_contrib)


# =========================================================
""" x1 = np.linspace(np.sqrt(sth+1e-1), np.sqrt(68)*kernels.mpi, 200)

y_teoLo = np.array(list(map(lambda s: ReT04(s**2), x1)))
y_ReggeHI = np.array(list(map(lambda s: Regge.Ret04(s**2), x1)))

y_teo = y_teoLo + y_ReggeHI
y_param = list(map(lambda s: t.t04_1(s**2).real, x1))

plt.plot(x1, y_teo, label="Roy")
plt.plot(x1, y_param, label="Parametrización")
plt.xlabel("$\\sqrt{s}$ (GeV)")
plt.ylabel("Re t04(s)")
plt.legend()
plt.show() """

raw = np.loadtxt("fort.40")

E = raw[:,0]/1000

KT_Ger = []
DT_Ger = []

for val in E:
    Ger = ReT04(val**2)
    KT_Ger.append(Ger[0])
    DT_Ger.append(Ger[1])

Regge_Ger = list(map(lambda s: Regge.Ret04(s**2), E))

with open("G0_Xray.txt", "w") as f:
    f.write("G0 WAVE CONTRIBUTIONS\n\n")
    f.write("-------------\n")
    f.write("Kernel contribs (S0 + P + S2)\n")
    for i in range(len(E)):
        f.write(f"{np.round(E[i], 4)}          {KT_Ger[i]}\n")

    f.write("\n-------------\n")
    f.write("Driving term contribs (D0 + G0 + F + D2 + G2)\n")
    for i in range(len(E)):
        f.write(f"{np.round(E[i], 4)}          {DT_Ger[i]}\n")
    
    f.write("\n-------------\n")
    f.write("Regge contrib\n")
    for i in range(len(E)):
        f.write(f"{np.round(E[i], 4)}          {Regge_Ger[i]}\n")
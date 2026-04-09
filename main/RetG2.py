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
DT_waves = [t.t02_1, t.t04_1, t.t13_1, t.t22_1] # D0, G0, F, D2

KT = [kernels.K2400, kernels.K2411, kernels.K2420] # S0, P, S2
DT = [kernels.K2402, kernels.K2404, kernels.K2413, kernels.K2422] # D0, G0, F, D2

# Interpolating parametrizations for shorter integration time

sp_grid = np.linspace(sth+eps, s2, 800)

# Diagonal (t24)
Im_t24_vals = np.array([t.t24_1(sp).imag for sp in sp_grid])
Im_t24_interp = PchipInterpolator(sp_grid, Im_t24_vals)

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


def ReT24(s):
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
    cauchy_kern = lambda sp: kernels.K2424_cauchy(s, sp) * Im_t24_interp(sp)
    cauchy_rest = lambda sp: kernels.K2424_rest(s, sp) * Im_t24_interp(sp)

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
    
    return KT_contrib+ DT_contrib


# =========================================================
x1 = np.linspace(np.sqrt(sth+1e-1), np.sqrt(68)*kernels.mpi, 200)

y_teoLo = np.array(list(map(lambda s: ReT24(s**2), x1)))
y_ReggeHI = np.array(list(map(lambda s: Regge.Ret24(s**2), x1)))

y_teo = y_teoLo + y_ReggeHI
y_param = list(map(lambda s: t.t24_1(s**2).real, x1))

plt.plot(x1, y_teo, label="Roy")
plt.plot(x1, y_param, label="Parametrización")
plt.xlabel("$\\sqrt{s}$ (GeV)")
plt.ylabel("Re t24(s)")
plt.legend()
plt.show()

""" raw = np.loadtxt("fort.40")

E = raw[:,0]/1000
ST_Jac = raw[:,1]
KT_Jac = raw[:,2]
DT_Jac = raw[:,3]
Regge_Jac = raw[:,4]

ST_Ger = []
KT_Ger = []
DT_Ger = []

for val in E:
    Ger = ReT00(val**2)
    ST_Ger.append(Ger[0])
    KT_Ger.append(Ger[1])
    DT_Ger.append(Ger[2])

Regge_Ger = list(map(lambda s: Regge.Ret00(s**2), E))

with open("S0_Xray.txt", "w") as f:
    f.write("S0 WAVE COMPARISON\n\n")
    f.write("-------------\n")
    f.write("Subtraction term\n")
    for i in range(len(E)):
        f.write(f"{np.round(E[i], 4)}          {ST_Jac[i]}          {ST_Ger[i]}\n")
    
    f.write("\n-------------\n")
    f.write("Kernel contribs (S0 + P + S2)\n")
    for i in range(len(E)):
        f.write(f"{np.round(E[i], 4)}          {KT_Jac[i]}          {KT_Ger[i]}\n")

    f.write("\n-------------\n")
    f.write("Driving term contribs (D0 + G0 + F + D2 + G2)\n")
    for i in range(len(E)):
        f.write(f"{np.round(E[i], 4)}          {DT_Jac[i]}          {DT_Ger[i]}\n")
    
    f.write("\n-------------\n")
    f.write("Regge contrib\n")
    for i in range(len(E)):
        f.write(f"{np.round(E[i], 4)}          {Regge_Jac[i]}          {Regge_Ger[i]}\n") """
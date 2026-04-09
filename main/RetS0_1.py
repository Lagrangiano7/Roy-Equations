import numpy as np
import scipy.integrate as integrate
import kernels
from kernels import sth, s2
import t
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from Regge import Regge

eps=1e-6

KT_waves = [t.t11_1, t.t20_1] # P, S2
DT_waves = [t.t02_1, t.t04_1, t.t13_1, t.t22_1, t.t24_1] # D0, G0, F, D2, G2
K_ord = kernels.getS0()
KT = [kernels.K0011, kernels.K0020] # P, S2
DT = [kernels.K0002, kernels.K0004, kernels.K0013, kernels.K0022, kernels.K0024] # D0, G0, F, D2, G2
STS0 = lambda s: kernels.ST_S0(s, kernels.a00_1, kernels.a20_1)

# Interpolating parametrizations for shorter integration time

sp_grid = np.linspace(sth+eps, s2, 400)

# Diagonal (t00)
Im_t00_vals = np.array([t.t00_1(sp).imag for sp in sp_grid])
Im_t00_interp = PchipInterpolator(sp_grid, Im_t00_vals)

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


def ReT00(s):
    # KT (S0, P, S2)
    cauchy_kern = lambda sp: kernels.K0000_cauchy(s, sp) * Im_t00_interp(sp)
    cauchy_rest = lambda sp: kernels.K0000_rest(s, sp) * Im_t00_interp(sp)

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
    
    return (STS0(s), KT_contrib, DT_contrib)


# =========================================================
x1 = np.linspace(np.sqrt(sth+1e-3), np.sqrt(68)*kernels.mpi, 600)

y_teoLo = np.array(list(map(lambda s: ReT00(s**2), x1)))
y_ReggeHI = np.array(list(map(lambda s: Regge.Ret00(s**2), x1)))

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
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
wave_ord = np.delete(t.waves_1, 5)
K_ord = kernels.getS2()
STS2 = lambda s: kernels.ST_S2(s, kernels.a00_1_t, kernels.a20_1_t)

N=10
NInterv = 20

x100_1, w100_1 = gaussxwab(N, sth+eps, 4/5*s2)

interv = np.linspace(4/5*s2, s2, NInterv)

xs = [x100_1]
ws = [w100_1]
h = s2 - 4/5*s2

for i in range(len(interv)):
    x, w = gaussxwab(N, 4/5*s2+h*i/NInterv, 4/5*s2 + h*(i+1)/NInterv)
    xs.append(x)
    ws.append(w)

# Interpolating parametrizations for shorter integration time

sp_grid = np.linspace(sth+eps, s2, 1500)

# Diagonal (t00)
Im_t20_vals = np.array([t.t20_1(sp).imag for sp in sp_grid])
Im_t20_interp = PchipInterpolator(sp_grid, Im_t20_vals)

# Ondas no diagonales
Im_interp = []
for i in range(len(K_ord)):
    vals = np.array([wave_ord[i](sp).imag for sp in sp_grid])
    Im_interp.append(PchipInterpolator(sp_grid, vals))

# =========================================================


def ReT20(s):
    # S0 -> S0
    cauchy_kern = lambda sp: kernels.K2020_cauchy(s, sp) * Im_t20_interp(sp)
    cauchy_rest = lambda sp: kernels.K2020_rest(s, sp) * Im_t20_interp(sp)

    total_diag = integrate.quad(
        cauchy_kern,
        sth+eps, s2,
        weight="cauchy",
        wvar=s,
        limit=200
    )[0]

    for i in range(len(xs)):
        for j in range(N):
            total_diag += ws[i][j]*cauchy_rest(xs[i][j])

    # Rest -> S0
    total_rest = 0
    for i in range(len(K_ord)):
        integrand = lambda sp: K_ord[i](s, sp) * Im_interp[i](sp)
        for j in range(len(xs)):
            for k in range(N):
                total_diag += ws[j][k]*integrand(xs[j][k])
    
    return STS2(s) + total_diag + total_rest


# =========================================================

x1 = np.linspace(np.sqrt(sth+1e-3), np.sqrt(68)*kernels.mpi, 600)

y_teoLo100 = np.array(list(map(lambda s: ReT20(s**2), x1)))
y_ReggeHI = np.array(list(map(lambda s: reader.getWave("S2")(s**2), x1)))

y_teo100 = y_teoLo100 + y_ReggeHI
y_param = list(map(lambda s: t.t20_1(s**2).real, x1))

raw = np.loadtxt("roys2.out")

x = raw[:,0]/1000
Ret00_Jacobo_disp = raw[:,1]
Ret00_Jacobo_param = raw[:,2]

plt.plot(x, Ret00_Jacobo_disp, label="Dispersivo Jacobo")


plt.plot(x1, y_teo100, label="Roy (N=200)")
plt.plot(x1, y_param, label="Parametrización")
plt.xlabel("$\\sqrt{s}$ (GeV)")
plt.ylabel("Re t11(s)")
plt.legend()
plt.show()

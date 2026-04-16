import numpy as np
import scipy.integrate as integrate
import kernels as k
from kernels import sth, s2
import t
import matplotlib.pyplot as plt
import util.interp as interp
from Regge import Regge

### Calculating Delta_0 = Ret11-Param11 without changing any parameters

eps=1e-6
N_pts = 50 # number of points we use to evaluate the function. More ==> smoother curve but higher execution time
                                                              # Less than 30 ==> fast but unreliable


# Interpolating parametrizations for shorter integration time

sp_grid = np.linspace(sth+eps, s2, 400)

# Calculating D0 = param11 - Ret11 (without changing any parameters)

x1 = np.linspace(np.sqrt(sth+1e-3), np.sqrt(68)*k.mpi, N_pts) # sqrt(s)

STP1 = np.array(list(map(lambda s: k.ST_P1(s**2, k.a00_1, k.a20_1), x1)))

# Evaluating by contributions in all 30 points of s grid
S0, D0, G0, P, F, S2, D2, G2 = [np.zeros(N_pts) for _ in range(8)]

# Ordinary (unaltered) interpolations

Im_t00_interp = interp.getInterp(t.t00_1, sp_grid)
Im_t02_interp = interp.getInterp(t.t02_1, sp_grid)
Im_t04_interp = interp.getInterp(t.t04_1, sp_grid)
Im_t11_interp = interp.getInterp(t.t11_1, sp_grid)
Im_t13_interp = interp.getInterp(t.t13_1, sp_grid)
Im_t20_interp = interp.getInterp(t.t20_1, sp_grid)
Im_t22_interp = interp.getInterp(t.t22_1, sp_grid)
Im_t24_interp = interp.getInterp(t.t24_1, sp_grid)

for (i, s) in enumerate(x1**2):
    S0_contrib = integrate.quad(lambda sp: k.K1100(s, sp)*Im_t00_interp(sp), sth+eps, s2, limit=200)[0]
    D0_contrib = integrate.quad(lambda sp: k.K1102(s, sp)*Im_t02_interp(sp), sth+eps, s2, limit=200)[0]
    G0_contrib = integrate.quad(lambda sp: k.K1104(s, sp)*Im_t04_interp(sp), sth+eps, s2, limit=200)[0]
    P_contrib = integrate.quad(lambda sp: k.K1111_cauchy(s, sp)*Im_t11_interp(sp), sth+eps, s2, weight="cauchy", wvar=s, limit=200)[0] + integrate.quad(lambda sp: k.K1111_rest(s, sp)*Im_t11_interp(sp), sth+eps, s2, limit=200)[0]
    F_contrib = integrate.quad(lambda sp: k.K1113(s, sp)*Im_t13_interp(sp), sth+eps, s2, limit=200)[0]
    S2_contrib = integrate.quad(lambda sp: k.K1120(s, sp)*Im_t20_interp(sp), sth+eps, s2, limit=200)[0]
    D2_contrib = integrate.quad(lambda sp: k.K1122(s, sp)*Im_t22_interp(sp), sth+eps, s2, limit=200)[0]
    G2_contrib = integrate.quad(lambda sp: k.K1124(s, sp)*Im_t24_interp(sp), sth+eps, s2, limit=200)[0]

    S0[i] = S0_contrib
    D0[i] = D0_contrib
    G0[i] = G0_contrib
    P[i] = P_contrib
    F[i] = F_contrib
    S2[i] = S2_contrib
    D2[i] = D2_contrib
    G2[i] = G2_contrib

Roy_original = STP1+S0+D0+G0+P+F+S2+D2+G2
Regge_original = np.array(list(map(lambda s: Regge.Ret11(s**2), x1)))

eval_original = Roy_original + Regge_original
param_original = np.array(list(map(lambda s: t.t11_1(s**2).real, x1)))

D_original = eval_original - param_original # Residue of Roy with respect to parametrization for all "s" values. No params changed.

### Beginning of varying parameters within 1sigma (up and down)

# Residues are stored in a matrix M such that:
  # Rows: parameter varied
  # Columns: Delta_i(s) = eval changing ith - param (changed if necessary)

M = np.zeros((72 + len(Regge.params_Regge), N_pts)) # Params from partial wave parametrizations + Regge parametrization

################ t00: recalculate only S0

counter = 0

par_s0_1 = t.par_s0_1.copy()
err_par_s0_1 = t.err_par_s0_1.copy()
S0_plus = np.zeros(N_pts)
S0_minus = np.zeros(N_pts)
STP1_plus = np.zeros(N_pts)
STP1_minus = np.zeros(N_pts)

eval_tochange = Roy_original - STP1 - S0 + Regge_original

# Change ST because it goes with a00 and a20 --> Ret00(sth) and Ret20(sth)

for i in range(len(par_s0_1)):
    print(i)
    # +1sigma in ith param.
    par_plus = par_s0_1.copy()
    par_plus[i] += err_par_s0_1[i]
    par_minus = par_s0_1.copy()
    par_minus[i] -= err_par_s0_1[i]

    t00_plus = lambda s: t.params1_s0.tf(s, par_plus)
    t00_minus = lambda s: t.params1_s0.tf(s, par_minus)

    a00_plus = t00_plus(sth+1e-6).real
    a00_minus = t00_minus(sth+1e-6).real

    Im_t00_plus_interp = interp.getInterp(t00_plus, sp_grid)
    Im_t00_minus_interp = interp.getInterp(t00_minus, sp_grid)

    for (j, s) in enumerate(x1**2):
        S0_plus[j] = integrate.quad(lambda sp: k.K1100(s, sp)*Im_t00_plus_interp(sp), sth+eps, s2, limit=200)[0]
        S0_minus[j] = integrate.quad(lambda sp: k.K1100(s, sp)*Im_t00_minus_interp(sp), sth+eps, s2, limit=200)[0]
        STP1_plus[j] = k.ST_P1(s, a00_plus, k.a20_1)
        STP1_minus[j] = k.ST_P1(s, a00_minus, k.a20_1)

    eval_plus = eval_tochange + S0_plus + STP1_plus
    eval_minus = eval_tochange + S0_minus + STP1_minus
    
    D_plus = eval_plus - param_original
    D_minus = eval_minus - param_original

    delta_plus = np.abs(D_plus-D_original)
    delta_minus = np.abs(D_minus-D_original)

    M[i,:] = np.maximum(delta_plus, delta_minus)

counter += len(par_s0_1)

################ t11: recalculate only P (diagonal)

par_p_1 = t.par_p1_1.copy()
err_par_p_1 = t.err_par_p1_1.copy()
P_plus = np.zeros(N_pts)
P_minus = np.zeros(N_pts)

eval_tochange = Roy_original - P + Regge_original

for i in range(len(par_p_1)):
    print(counter+i)
    # +1sigma in ith param.
    par_plus = par_p_1.copy()
    par_plus[i] += err_par_p_1[i]
    par_minus = par_p_1.copy()
    par_minus[i] -= err_par_p_1[i]

    t11_plus = lambda s: t.params1_p1.tf(s, par_plus)
    t11_minus = lambda s: t.params1_p1.tf(s, par_minus)
    Im_t11_plus_interp = interp.getInterp(t11_plus, sp_grid)
    Im_t11_minus_interp = interp.getInterp(t11_minus, sp_grid)

    for (j, s) in enumerate(x1**2):
        P_plus[j] = integrate.quad(lambda sp: k.K1111_cauchy(s, sp)*Im_t11_plus_interp(sp), sth+eps, s2, weight="cauchy", wvar=s, limit=200)[0] + integrate.quad(lambda sp: k.K1111_rest(s, sp)*Im_t11_plus_interp(sp), sth+eps, s2, limit=200)[0]
        P_minus[j] = integrate.quad(lambda sp: k.K1111_cauchy(s, sp)*Im_t11_minus_interp(sp), sth+eps, s2, weight="cauchy", wvar=s, limit=200)[0] + integrate.quad(lambda sp: k.K1111_rest(s, sp)*Im_t11_minus_interp(sp), sth+eps, s2, limit=200)[0]

    eval_plus = eval_tochange + P_plus
    eval_minus = eval_tochange + P_minus
    
    D_plus = eval_plus - [t11_plus(s).real for s in x1**2] # New param with +1sigma
    D_minus = eval_minus - [t11_minus(s).real for s in x1**2] # New param with -1sigma

    delta_plus = np.abs(D_plus-D_original)
    delta_minus = np.abs(D_minus-D_original)

    M[i+counter,:] = np.maximum(delta_plus, delta_minus)

counter += len(par_p_1)

################ t02: recalculate only D0

par_d0_1 = t.par_d0_1.copy()
err_par_d0_1 = t.err_par_d0_1.copy()
D0_plus = np.zeros(N_pts)
D0_minus = np.zeros(N_pts)

eval_tochange = Roy_original - D0 + Regge_original

for i in range(len(par_d0_1)):
    print(counter+i)
    # +1sigma in ith param.
    par_plus = par_d0_1.copy()
    par_plus[i] += err_par_d0_1[i]
    par_minus = par_d0_1.copy()
    par_minus[i] -= err_par_d0_1[i]

    Im_t02_plus_interp = interp.getInterp(lambda s: t.params1_d0.tf(s, par_plus), sp_grid)
    Im_t02_minus_interp = interp.getInterp(lambda s: t.params1_d0.tf(s, par_minus), sp_grid)

    for (j, s) in enumerate(x1**2):
        D0_plus[j] = integrate.quad(lambda sp: k.K1102(s, sp)*Im_t02_plus_interp(sp), sth+eps, s2, limit=200)[0]
        D0_minus[j] = integrate.quad(lambda sp: k.K1102(s, sp)*Im_t02_minus_interp(sp), sth+eps, s2, limit=200)[0]

    eval_plus = eval_tochange + D0_plus
    eval_minus = eval_tochange + D0_minus
    
    D_plus = eval_plus - param_original
    D_minus = eval_minus - param_original

    delta_plus = np.abs(D_plus-D_original)
    delta_minus = np.abs(D_minus-D_original)

    M[i+counter,:] = np.maximum(delta_plus, delta_minus)

counter += len(par_d0_1)

################ t04: recalculate only G0

par_g0_1 = t.par_g0_1.copy()
err_par_g0_1 = t.err_par_g0_1.copy()
G0_plus = np.zeros(N_pts)
G0_minus = np.zeros(N_pts)

eval_tochange = Roy_original - G0 + Regge_original

for i in range(len(par_g0_1)):
    print(counter+i)
    # +1sigma in ith param.
    par_plus = par_g0_1.copy()
    par_plus[i] += err_par_g0_1[i]
    par_minus = par_g0_1.copy()
    par_minus[i] -= err_par_g0_1[i]

    Im_t04_plus_interp = interp.getInterp(lambda s: t.params1_g0.tf(s, par_plus), sp_grid)
    Im_t04_minus_interp = interp.getInterp(lambda s: t.params1_g0.tf(s, par_minus), sp_grid)

    for (j, s) in enumerate(x1**2):
        G0_plus[j] = integrate.quad(lambda sp: k.K1104(s, sp)*Im_t04_plus_interp(sp), sth+eps, s2, limit=200)[0]
        G0_minus[j] = integrate.quad(lambda sp: k.K1104(s, sp)*Im_t04_minus_interp(sp), sth+eps, s2, limit=200)[0]

    eval_plus = eval_tochange + G0_plus
    eval_minus = eval_tochange + G0_minus
    
    D_plus = eval_plus - param_original
    D_minus = eval_minus - param_original

    delta_plus = np.abs(D_plus-D_original)
    delta_minus = np.abs(D_minus-D_original)

    M[i+counter,:] = np.maximum(delta_plus, delta_minus)

counter += len(par_g0_1)

################ t13: recalculate only F

par_f_1 = t.par_f1_1.copy()
err_par_f_1 = t.err_par_f1_1.copy()
F_plus = np.zeros(N_pts)
F_minus = np.zeros(N_pts)

eval_tochange = Roy_original - F + Regge_original

for i in range(len(par_f_1)):
    print(counter+i)
    # +1sigma in ith param.
    par_plus = par_f_1.copy()
    par_plus[i] += err_par_f_1[i]
    par_minus = par_f_1.copy()
    par_minus[i] -= err_par_f_1[i]

    Im_t13_plus_interp = interp.getInterp(lambda s: t.params1_f1.tf(s, par_plus), sp_grid)
    Im_t13_minus_interp = interp.getInterp(lambda s: t.params1_f1.tf(s, par_minus), sp_grid)

    for (j, s) in enumerate(x1**2):
        F_plus[j] = integrate.quad(lambda sp: k.K1113(s, sp)*Im_t13_plus_interp(sp), sth+eps, s2, limit=200)[0]
        F_minus[j] = integrate.quad(lambda sp: k.K1113(s, sp)*Im_t13_minus_interp(sp), sth+eps, s2, limit=200)[0]

    eval_plus = eval_tochange + F_plus
    eval_minus = eval_tochange + F_minus
    
    D_plus = eval_plus - param_original
    D_minus = eval_minus - param_original

    delta_plus = np.abs(D_plus-D_original)
    delta_minus = np.abs(D_minus-D_original)

    M[i+counter,:] = np.maximum(delta_plus, delta_minus)

counter += len(par_f_1)

################ t20: recalculate only S2

par_s2_1 = t.par_s2_1.copy()
err_par_s2_1 = t.err_par_s2_1.copy()
S2_plus = np.zeros(N_pts)
S2_minus = np.zeros(N_pts)
STP1_plus = np.zeros(N_pts)
STP1_minus = np.zeros(N_pts)

eval_tochange = Roy_original - STP1 - S2 + Regge_original

# Change ST because it goes with a00 and a20 --> Ret00(sth) and Ret20(sth)

for i in range(len(par_s2_1)):
    print(counter+i)
    # +1sigma in ith param.
    par_plus = par_s2_1.copy()
    par_plus[i] += err_par_s2_1[i]
    par_minus = par_s2_1.copy()
    par_minus[i] -= err_par_s2_1[i]

    t20_plus = lambda s: t.params1_s2.tf(s, par_plus)
    t20_minus = lambda s: t.params1_s2.tf(s, par_minus)

    a20_plus = t20_plus(sth+1e-6).real
    a20_minus = t20_minus(sth+1e-6).real

    Im_t20_plus_interp = interp.getInterp(t20_plus, sp_grid)
    Im_t20_minus_interp = interp.getInterp(t20_minus, sp_grid)

    for (j, s) in enumerate(x1**2):
        S2_plus[j] = integrate.quad(lambda sp: k.K1120(s, sp)*Im_t20_plus_interp(sp), sth+eps, s2, limit=200)[0]
        S2_minus[j] = integrate.quad(lambda sp: k.K1120(s, sp)*Im_t20_minus_interp(sp), sth+eps, s2, limit=200)[0]
        STP1_plus[j] = k.ST_P1(s, k.a00_1, a20_plus)
        STP1_minus[j] = k.ST_P1(s, k.a00_1, a20_minus)

    eval_plus = eval_tochange + S2_plus + STP1_plus
    eval_minus = eval_tochange + S2_minus + STP1_minus
    
    D_plus = eval_plus - param_original
    D_minus = eval_minus - param_original

    delta_plus = np.abs(D_plus-D_original)
    delta_minus = np.abs(D_minus-D_original)

    M[i+counter,:] = np.maximum(delta_plus, delta_minus)

counter += len(par_s2_1)

################ t22: recalculate only D2

par_d2_1 = t.par_d2_1.copy()
err_par_d2_1 = t.err_par_d2_1.copy()
D2_plus = np.zeros(N_pts)
D2_minus = np.zeros(N_pts)

eval_tochange = Roy_original - D2 + Regge_original

for i in range(len(par_d2_1)):
    print(counter+i)
    # +1sigma in ith param.
    par_plus = par_d2_1.copy()
    par_plus[i] += err_par_d2_1[i]
    par_minus = par_d2_1.copy()
    par_minus[i] -= err_par_d2_1[i]

    Im_t22_plus_interp = interp.getInterp(lambda s: t.params1_d2.tf(s, par_plus), sp_grid)
    Im_t22_minus_interp = interp.getInterp(lambda s: t.params1_d2.tf(s, par_minus), sp_grid)

    for (j, s) in enumerate(x1**2):
        D2_plus[j] = integrate.quad(lambda sp: k.K1122(s, sp)*Im_t22_plus_interp(sp), sth+eps, s2, limit=200)[0]
        D2_minus[j] = integrate.quad(lambda sp: k.K1122(s, sp)*Im_t22_minus_interp(sp), sth+eps, s2, limit=200)[0]

    eval_plus = eval_tochange + D2_plus
    eval_minus = eval_tochange + D2_minus
    
    D_plus = eval_plus - param_original
    D_minus = eval_minus - param_original

    delta_plus = np.abs(D_plus-D_original)
    delta_minus = np.abs(D_minus-D_original)

    M[i+counter,:] = np.maximum(delta_plus, delta_minus)

counter += len(par_d2_1)

################ t24: recalculate only G2

par_g2_1 = t.par_g2_1.copy()
err_par_g2_1 = t.err_par_g2_1.copy()
G2_plus = np.zeros(N_pts)
G2_minus = np.zeros(N_pts)

eval_tochange = Roy_original - G2 + Regge_original

for i in range(len(par_g2_1)):
    print(counter+i)
    # +1sigma in ith param.
    par_plus = par_g2_1.copy()
    par_plus[i] += err_par_g2_1[i]
    par_minus = par_g2_1.copy()
    par_minus[i] -= err_par_g2_1[i]

    Im_t24_plus_interp = interp.getInterp(lambda s: t.params1_g2.tf(s, par_plus), sp_grid)
    Im_t24_minus_interp = interp.getInterp(lambda s: t.params1_g2.tf(s, par_minus), sp_grid)

    for (j, s) in enumerate(x1**2):
        G2_plus[j] = integrate.quad(lambda sp: k.K1124(s, sp)*Im_t24_plus_interp(sp), sth+eps, s2, limit=200)[0]
        G2_minus[j] = integrate.quad(lambda sp: k.K1124(s, sp)*Im_t24_minus_interp(sp), sth+eps, s2, limit=200)[0]

    eval_plus = eval_tochange + G2_plus
    eval_minus = eval_tochange + G2_minus
    
    D_plus = eval_plus - param_original
    D_minus = eval_minus - param_original

    delta_plus = np.abs(D_plus-D_original)
    delta_minus = np.abs(D_minus-D_original)

    M[i+counter,:] = np.maximum(delta_plus, delta_minus)

counter += len(par_g2_1)

############################### Regge ###############################

Regge_plus = np.zeros(N_pts)
Regge_minus = np.zeros(N_pts)

eval_tochange = Roy_original # then I add modified Regges

for tag in Regge.params_Regge.keys():
    print(counter+1)
    params_Regge_plus = Regge.params_Regge.copy()
    params_Regge_minus = Regge.params_Regge.copy()

    params_Regge_plus[tag] += Regge.errs_Regge[tag]
    params_Regge_minus[tag] -= Regge.errs_Regge[tag]

    for (j, s) in enumerate(x1**2):
        Regge_plus[j] = Regge.Ret11(s, params_Regge_plus)
        Regge_minus[j] = Regge.Ret11(s, params_Regge_minus)
    
    eval_plus = eval_tochange + Regge_plus
    eval_minus = eval_tochange + Regge_minus
    
    D_plus = eval_plus - param_original
    D_minus = eval_minus - param_original

    delta_plus = np.abs(D_plus-D_original)
    delta_minus = np.abs(D_minus-D_original)

    M[counter,:] = np.maximum(delta_plus, delta_minus)
    counter += 1

print(np.where(M==0.0))
####### Propagating error

errs = np.sqrt(np.sum(M**2, axis=0))

up = eval_original + errs
down = eval_original - errs

np.savetxt("P_bands.txt", np.column_stack([x1, eval_original, [t.t11_1(val).real for val in x1**2] + errs, [t.t11_1(val).real for val in x1**2] - errs]))

# Plotting with error band

plt.plot(x1, eval_original, label="Roy", color="blue")
plt.plot(x1, param_original, label="Parametrization", color="red")

plt.fill_between(
    x1,
    down,
    up,
    color="blue",
    alpha=0.3,
    label="Roy error band"
)

plt.xlabel("$\\sqrt{s}$ [GeV]")
plt.ylabel("Re t11(s)")

plt.legend()
plt.show()
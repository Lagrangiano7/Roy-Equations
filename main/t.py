# All partial wave parametrizations

import numpy as np
from kernels import sth
import matplotlib.pyplot as plt

import params.s0_global1_opt as params1_s0
import params.s2_new_b as params1_s2
import params.p_global_mart_new_1_rewritten as params1_p1
import params.d0_900_1_try as params1_d0
import params.d2_pw_try as params1_d2
import params.f_new_1_unit as params1_f1
import params.g0_unit_new as params1_g0
import params.g2_pw_try as params1_g2

raw = np.loadtxt("params/monteKmod_sol1_def.in", comments="|")
par = raw[:,0]
err_par = raw[:,1]


# Selecting indices to slice global parameter array
par_s0_in=list(range(14))
par_s2_in=list(range(32,32+len(params1_s2.par)))
par_p1_in=list(range(14,14+len(params1_p1.par)))
par_d0_in=list(range(38,38+len(params1_d0.par)))
par_d2_in=list(range(50,50+len(params1_d2.par)))
par_f1_in=list(range(56,56+len(params1_f1.par)))
par_g0_in=list(range(62,62+len(params1_g0.par)))
par_g2_in=list(range(66,66+len(params1_g2.par)))

# Slicing: selecting the parameters corresponding to each partial wave up to G-wave
par_s0_1=np.take(par, par_s0_in)
par_s2_1=np.take(par, par_s2_in)
par_p1_1 = np.take(par, par_p1_in)
par_d0_1=np.take(par, par_d0_in)
par_d2_1=np.take(par, par_d2_in)
par_f1_1=np.take(par, par_f1_in)
par_g0_1=np.take(par, par_g0_in)
par_g2_1=np.take(par, par_g2_in)

err_par_s0_1=np.take(err_par, par_s0_in)
err_par_s2_1=np.take(err_par, par_s2_in)
err_par_p1_1 = np.take(err_par, par_p1_in)
err_par_d0_1=np.take(err_par, par_d0_in)
err_par_d2_1=np.take(err_par, par_d2_in)
err_par_f1_1=np.take(err_par, par_f1_in)
err_par_g0_1=np.take(err_par, par_g0_in)
err_par_g2_1=np.take(err_par, par_g2_in)

# Building parametrizations from first param. set

t00_1 = lambda s: params1_s0.tf(s, par_s0_1) # S0
t20_1 = lambda s: params1_s2.tf(s, par_s2_1) # S2
t11_1 = lambda s: params1_p1.tf(s, par_p1_1) # P1
t02_1 = lambda s: params1_d0.tf(s, par_d0_1) # D0
t22_1 = lambda s: params1_d2.tf(s, par_d2_1) # D2
t13_1 = lambda s: params1_f1.tf(s, par_f1_1) # F1
t04_1 = lambda s: params1_g0.tf(s, par_g0_1) # G0
t24_1 = lambda s: params1_g2.tf(s, par_g2_1) # G2
# K0002, K0004, K0011, K0013, K0020, K0022, K0024
# D0, G0, P1, F1, S2, D2, G2

waves_1 = np.array([t00_1, t02_1, t04_1, t11_1, t13_1, t20_1, t22_1, t24_1]) # S0, D0, G0, P1, F1, S2, D2, G2

""" import params.s0_global2_opt as params2_s0
import params.s2_new_b as params2_s2
import params.p_global_mart_new_2_rewritten as params2_p1
import params.d0_kk_2_try as params2_d0
import params.d2_pw_try as params2_d2
import params.f_new_2_unit as params2_f1
import params.g0_unit_new as params2_g0
import params.g2_pw_try as params2_g2

f=open('params/monteKmod_sol2_def.in',"r") # Params from constrained
lines=f.readlines()
par=[] # Exctract all params in file (CONSTRAINED)
for x in lines:
    par.append(float(x.split()[0]))
f.close()


# Selecting indices to slice global parameter array
par_s0_in=list(range(len(params2_s0.par)))
par_s2_in=list(range(31,31+len(params2_s2.par)))
par_p1_in=list(range(16,16+len(params2_p1.par)))
par_d0_in=list(range(37,37+len(params2_d0.par)))
par_d2_in=list(range(49,49+len(params2_d2.par)))
par_f1_in=list(range(55,55+len(params2_f1.par)))
par_g0_in=list(range(61,61+len(params2_g0.par)))
par_g2_in=list(range(65,65+len(params2_g2.par)))

# Slicing: selecting the parameters corresponding to each partial wave up to G-wave
par_s0_2=np.take(par, par_s0_in)
par_s2_2=np.take(par, par_s2_in)
par_p1_2 = np.take(par, par_p1_in)
par_d0_2=np.take(par, par_d0_in)
par_d2_2=np.take(par, par_d2_in)
par_f1_2=np.take(par, par_f1_in)
par_g0_2=np.take(par, par_g0_in)
par_g2_2=np.take(par, par_g2_in)

t00_2 = lambda s: params2_s0.tf(s**2, par_s0_2)
t20_2 = lambda s: params2_s2.tf(s**2, par_s2_2)
t11_2 = lambda s: params2_p1.tf(s**2, par_p1_2)
t02_2 = lambda s: params2_d0.tf(s**2, par_d0_2)
t22_2 = lambda s: params2_d2.tf(s**2, par_d2_2)
t13_2 = lambda s: params2_f1.tf(s**2, par_f1_2)
t04_2 = lambda s: params2_g0.tf(s**2, par_g0_2)
t24_2 = lambda s: params2_g2.tf(s**2, par_g2_2)
"""
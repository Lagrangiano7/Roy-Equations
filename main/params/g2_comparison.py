#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:58:09 2023

@author: pablo
"""

import matplotlib.pyplot as plt
import numpy as np
import g2_pw_try as p1
import g2_pw_try as p2
import g2_pw_try as p3

f=open('monteKmod_sol1_def.in',"r")
lines=f.readlines()
par=[]
for x in lines:
    par.append(float(x.split()[0]))
f.close()


n_s0=66
n_p=len(p1.par)
par_p1_in=list(range(n_s0,n_s0+n_p))
par_p1=np.take(par, par_p1_in)


f=open('monteKmod_sol2_def.in',"r")
lines=f.readlines()
par=[]
for x in lines:
    par.append(float(x.split()[0]))
f.close()


n_s0=65
n_p=len(p2.par)
par_p2_in=list(range(n_s0,n_s0+n_p))
par_p2=np.take(par, par_p2_in)

f=open('monteKmod_sol3_def.in',"r")
lines=f.readlines()
par=[]
for x in lines:
    par.append(float(x.split()[0]))
f.close()


n_s0=67
n_p=len(p3.par)
par_p3_in=list(range(n_s0,n_s0+n_p))
par_p3=np.take(par, par_p3_in)

"""Paper"""

fig, ax = plt.subplots(figsize=(16, 12), dpi=100, sharex=True)

x=np.linspace(0.28,2.2,200)
y=np.array(list(map(lambda z: p1.deltaf2(z**2,par_p1), x)))
y2=np.array(list(map(lambda z: p1.deltaf2(z**2,p1.par), x)))
erru=list(map(lambda z: p1.errdu(z**2,par_p1,p1.err_par), x))
erru2=list(map(lambda z: p1.errdu(z**2,p1.par,p1.err_par), x))
errd=list(map(lambda z: p1.errdd(z**2,par_p1,p1.err_par), x))
errd2=list(map(lambda z: p1.errdd(z**2,p1.par,p1.err_par), x))

y_2=np.array(list(map(lambda z: p2.deltaf2(z**2,par_p2), x)))
y2_2=np.array(list(map(lambda z: p2.deltaf2(z**2,p2.par), x)))
erru_2=list(map(lambda z: p2.errdu(z**2,par_p2,p2.err_par), x))
erru2_2=list(map(lambda z: p2.errdu(z**2,p2.par,p2.err_par), x))
errd_2=list(map(lambda z: p2.errdd(z**2,par_p2,p2.err_par), x))
errd2_2=list(map(lambda z: p2.errdd(z**2,p2.par,p2.err_par), x))

y_3=np.array(list(map(lambda z: p3.deltaf2(z**2,par_p3), x)))
y2_3=np.array(list(map(lambda z: p3.deltaf2(z**2,p3.par), x)))
erru_3=list(map(lambda z: p3.errdu(z**2,par_p3,p3.err_par), x))
erru2_3=list(map(lambda z: p3.errdu(z**2,p3.par,p3.err_par), x))
errd_3=list(map(lambda z: p3.errdd(z**2,par_p3,p3.err_par), x))
errd2_3=list(map(lambda z: p3.errdd(z**2,p3.par,p3.err_par), x))


plt.fill_between(x, y+errd, y+erru,color='k',alpha=0.4,zorder=0)
plt.fill_between(x, y_2+errd_2, y_2+erru_2,color='tab:red',alpha=0.4,zorder=0)
plt.fill_between(x, y_3+errd_3, y_3+erru_3,color='tab:cyan',alpha=0.5,zorder=0)
plt.fill_between([1.6,2.15], [-37,-37],[2,2],color='k',alpha=0.2,zorder=0)
plt.plot([0.3,2.15],[0,0],'--',color='grey',zorder=0)
plt.plot(x,y,'k',label='Global I',zorder=11,linewidth=3)
plt.plot(x,y_2,'tab:red',label='Global II',zorder=10,linewidth=3)
plt.plot(x,y_3,'dodgerblue',label='Global III',zorder=10,linewidth=3)
plt.xlabel('$\sqrt{s}$ (GeV)',fontsize=30)
plt.xlim([0.28,2.15])
plt.ylim([-7,0.5])
plt.tick_params(axis='x',labelsize=23)
plt.tick_params(axis='y',labelsize=23)
plt.legend(fontsize=23) 
plt.text(0.36,-3, '$\delta^{(2)}_4(s)$ ($^{\circ}$)',fontsize=48)
plt.tick_params(right=True, left=True, axis='y', color='k', length=12,
                        grid_color='none',direction='inout')
plt.tick_params(bottom=True, top=True,axis='x', color='k', length=12,
                        grid_color='none',direction='inout')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:58:09 2023

@author: pablo
"""

import matplotlib.pyplot as plt
import numpy as np
import f_new_1_unit as p1
import f_new_2_unit as p2
import f_new_3_unit as p3

f=open('monteKmod_sol1_def.in',"r")
lines=f.readlines()
par=[]
for x in lines:
    par.append(float(x.split()[0]))
f.close()


n_s0=56
n_p=len(p1.par)
par_p1_in=list(range(n_s0,n_s0+n_p))
par_p1=np.take(par, par_p1_in)


f=open('monteKmod_sol2_def.in',"r")
lines=f.readlines()
par=[]
for x in lines:
    par.append(float(x.split()[0]))
f.close()


n_s0=55
n_p=len(p2.par)
par_p2_in=list(range(n_s0,n_s0+n_p))
par_p2=np.take(par, par_p2_in)

f=open('monteKmod_sol3_def.in',"r")
lines=f.readlines()
par=[]
for x in lines:
    par.append(float(x.split()[0]))
f.close()


n_s0=57
n_p=len(p3.par)
par_p3_in=list(range(n_s0,n_s0+n_p))
par_p3=np.take(par, par_p3_in)


"""Paper"""

fig, ax = plt.subplots(2,figsize=(16, 24), dpi=100, sharex=True)

x=np.linspace(0.28,1.8,200)
x1=np.linspace(0.28,1.9,200)
y=np.array(list(map(lambda z: p1.deltaf(z**2,par_p1), x1)))
y2=np.array(list(map(lambda z: p1.deltaf(z**2,p1.par), x1)))
erru=list(map(lambda z: p1.errdu(z**2,par_p1,p1.err_par), x1))
erru2=list(map(lambda z: p1.errdu(z**2,p1.par,p1.err_par), x1))
errd=list(map(lambda z: p1.errdd(z**2,par_p1,p1.err_par), x1))
errd2=list(map(lambda z: p1.errdd(z**2,p1.par,p1.err_par), x1))

y_2=np.array(list(map(lambda z: p2.deltaf(z**2,par_p2), x)))
y2_2=np.array(list(map(lambda z: p2.deltaf(z**2,p2.par), x)))
erru_2=list(map(lambda z: p2.errdu(z**2,par_p2,p2.err_par), x))
erru2_2=list(map(lambda z: p2.errdu(z**2,p2.par,p2.err_par), x))
errd_2=list(map(lambda z: p2.errdd(z**2,par_p2,p2.err_par), x))
errd2_2=list(map(lambda z: p2.errdd(z**2,p2.par,p2.err_par), x))

y_3=np.array(list(map(lambda z: p3.deltaf(z**2,par_p3), x)))
y2_3=np.array(list(map(lambda z: p3.deltaf(z**2,p3.par), x)))
erru_3=list(map(lambda z: p3.errdu(z**2,par_p3,p3.err_par), x))
erru2_3=list(map(lambda z: p3.errdu(z**2,p3.par,p3.err_par), x))
errd_3=list(map(lambda z: p3.errdd(z**2,par_p3,p3.err_par), x))
errd2_3=list(map(lambda z: p3.errdd(z**2,p3.par,p3.err_par), x))


ax[0].fill_between(x1, y+errd, y+erru,color='k',alpha=0.4,zorder=0)
ax[0].fill_between(x, y_2+errd_2, y_2+erru_2,color='tab:red',alpha=0.4,zorder=0)
ax[0].fill_between(x, y_3+errd_3, y_3+erru_3,color='tab:cyan',alpha=0.5,zorder=0)
ax[0].fill_between([1.6,1.9], [-14,-14],[14,14],color='k',alpha=0.2,zorder=0)
ax[0].plot([0.3,2],[0,0],'--',color='grey',zorder=0)
ax[0].plot(x1,y,'k',label='Global I',zorder=11,linewidth=3)
ax[0].plot(x,y_2,'tab:red',label='Global II',zorder=10,linewidth=3)
ax[0].plot(x,y_3,'dodgerblue',label='Global III',zorder=10,linewidth=3)
ax[0].set_xlabel('$\sqrt{s}$ (GeV)',fontsize=30)
ax[0].set_xlim([0.28,1.9])
ax[0].set_ylim([-14,14])
ax[0].tick_params(axis='x',labelsize=23)
ax[0].tick_params(axis='y',labelsize=23)
ax[0].legend(fontsize=23,loc=3) 
ax[0].text(0.5, 7, '$\delta^{(1)}_3(s)$ ($^{\circ}$)',fontsize=48)
ax[0].tick_params(right=True, left=True, axis='y', color='k', length=12,
                        grid_color='none',direction='inout')
ax[0].tick_params(bottom=True, top=True,axis='x', color='k', length=12,
                        grid_color='none',direction='inout')

y=np.array(list(map(lambda z: p1.etaf(z**2,par_p1), x1)))
y2=np.array(list(map(lambda z: p1.etaf(z**2,p1.par), x1)))
erru=list(map(lambda z: p1.erreu(z**2,par_p1,p1.err_par), x1))
erru2=list(map(lambda z: p1.erreu(z**2,p1.par,p1.err_par), x1))
errd=list(map(lambda z: p1.erred(z**2,par_p1,p1.err_par), x1))
errd2=list(map(lambda z: p1.erred(z**2,p1.par,p1.err_par), x1))

y_2=np.array(list(map(lambda z: p2.etaf(z**2,par_p2), x)))
y2_2=np.array(list(map(lambda z: p2.etaf(z**2,p2.par), x)))
erru_2=list(map(lambda z: p2.erreu(z**2,par_p2,p2.err_par), x))
erru2_2=list(map(lambda z: p2.erreu(z**2,p2.par,p2.err_par), x))
errd_2=list(map(lambda z: p2.erred(z**2,par_p2,p2.err_par), x))
errd2_2=list(map(lambda z: p2.erred(z**2,p2.par,p2.err_par), x))

y_3=np.array(list(map(lambda z: p3.etaf(z**2,par_p3), x)))
y2_3=np.array(list(map(lambda z: p3.etaf(z**2,p3.par), x)))
erru_3=list(map(lambda z: p3.erreu(z**2,par_p3,p3.err_par), x))
erru2_3=list(map(lambda z: p3.erreu(z**2,p3.par,p3.err_par), x))
errd_3=list(map(lambda z: p3.erred(z**2,par_p3,p3.err_par), x))
errd2_3=list(map(lambda z: p3.erred(z**2,p3.par,p3.err_par), x))


ax[1].fill_between(x1,y+ errd, y+erru,color='k',alpha=0.4,zorder=0)
ax[1].fill_between(x, y_2+ errd_2, y_2+erru_2,color='tab:red',alpha=0.4,zorder=0)
ax[1].fill_between(x, y_3+errd_3, y_3+erru_3,color='tab:cyan',alpha=0.5,zorder=0)
ax[1].fill_between([1.6,1.9], [0,0],[1.05,1.05],color='k',alpha=0.2,zorder=0)
ax[1].plot([0.3,2],[1,1],'--',color='grey',zorder=0)
ax[1].plot(x1,y,'k',label='Global I',zorder=11,linewidth=3)
ax[1].plot(x,y_2,'tab:red',label='Global II',zorder=10,linewidth=3)
ax[1].plot(x,y_3,'dodgerblue',label='Global III',zorder=10,linewidth=3)
ax[1].set_xlabel('$\sqrt{s}$ (GeV)',fontsize=30)
ax[1].set_ylim([0.0,1.05])
ax[1].tick_params(axis='x',labelsize=23)
ax[1].tick_params(axis='y',labelsize=23)
ax[1].legend(fontsize=23) 
ax[1].text(0.47, 0.7, '$\eta^{(1)}_3(s)$',fontsize=48)
ax[1].tick_params(right=True, left=True, axis='y', color='k', length=12,
                        grid_color='none',direction='inout')
ax[1].tick_params(bottom=True, top=True,axis='x', color='k', length=12,
                        grid_color='none',direction='inout')
plt.subplots_adjust(hspace=.0)
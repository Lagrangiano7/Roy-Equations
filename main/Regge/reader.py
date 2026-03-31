import numpy as np
from scipy.interpolate import interp1d

waveKey = {"S0":"00", "D0":"02", "G0":"04", "P1":"11", "F1":"11", "S2":"20", "D2":"22", "G2":"24"}

def getWave(name):
    data = np.loadtxt(f"Regge/contrib{waveKey[name]}_highenergy.csv", delimiter=",")
    s_vals = data[:,0]
    contrib_vals = data[:,1]

    # Crear interpolador (lineal o cúbico)
    contrib_interp = interp1d(s_vals, contrib_vals, kind='cubic', fill_value="extrapolate")
    return contrib_interp
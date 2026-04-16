from scipy.interpolate import PchipInterpolator
import numpy as np

def getInterp(f, axis):
    vals = np.array([f(sp).imag for sp in axis])
    return PchipInterpolator(axis, vals)
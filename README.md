# Introduction

This project is the codebase for my final degree project, "dispersive methods for the description of pion-pion interactions." It contains the numerical implementation of the Roy equations up to G-wave employing the parametrizations of Peláez et. al. for the partial waves conforming the full amplitude of pion-pion processes.

# Structure

The project is divided into three main functional parts:

1. The parametrizations of the partial waves
2. The calculation of the kernel and higher-order terms of the Roy equations up to the matching point
3. The implementation of the high-energy tail by means of the Regge trajectories

For completion, the "Mathematica" folder contains the analytic calculation of all 64 kernels employed in this project, as well as the subtraction terms

# Procedure

Firstly, all kernels and subtraction terms are calculated analytically employing Mathematica, and they are then transferred to the kernels.py file. Then, the low-energy contributions (i.e. the Roy equations up to the matching point for $J\leq 4$) are calculated, one at a time so that we can vary every parameter relevant to the wave at hand. These contributions are the real part we would expect the parametrizations to have acccording to the Roy equations, and they are obtained using their imaginary part as input. These variations are then stored in the matrix $M$. The entry of the matrix for each set of these indices is precisely the error associated to the partial wave at hand in the j-th $s$-point when varying the $i$-th parameter.

The Regge folder contains the numerical implementation of the high-energy tail, by taking the Regge trajectories of Peláez et. al., finding their real part from a full-amplitude dispersion relation and then extracting the desired partial wave from them. These trajectories are parametrizations and, as such, also carry error. This is propagated to the final result in the same fashion as before.

Finally, we may take the error at each $s$-point to be the quadratic sum of all the errors at said point, thus defining an error band for the difference between the dispersive result (i.e. the real part the parametrizations should have according to the Roy equations) and the parametrization's actual real part. All the final results may be accessed in the "img" folder.

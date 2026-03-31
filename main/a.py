import numpy as np
import matplotlib.pyplot as plt

from kernels import ST_S2, sth, s2, a00_1, a20_1, a00_1_t, a20_1_t, mpi

x = np.linspace(np.sqrt(sth+1e-3), np.sqrt(68)*mpi, 100)
plt.plot(x, ST_S2(x**2, a00_1, a20_1))
plt.plot(x, ST_S2(x**2, a00_1_t, a20_1_t))

plt.show()
import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

num_samples = 1
prefix = 'ns_'
title = "Energy landscape for No-slip case"

for j in range(num_samples):
    data = np.loadtxt(f'{prefix}energy_vals_{j}.txt', delimiter=' ')
    x.append(np.copy(data[:, 0]))
    y.append(np.copy(data[:, 1]/np.max(data[:, 1])))

for j in range(num_samples):
    xi = x[j]
    yi = y[j]
    plt.plot(xi, yi, label=str(j), marker='*')
plt.title(title)
plt.legend()
# plt.show()
plt.savefig('/u/s/g/sgsrinivasa2/Desktop/NSEL.png')


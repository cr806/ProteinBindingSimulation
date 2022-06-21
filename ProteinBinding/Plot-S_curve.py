import numpy as np
import matplotlib.pyplot as plt
from Settings import ON, OFF

folder = ['02-000085', '03-000085', '05-000085', '06-000085', '08-000085', '09-000085']
results = []
for f in folder:
    fname = f'OLD RESULTS/Population_OnRate/{f}/S-Curve_Pop100-5000.csv'
    results.append(np.loadtxt(fname, dtype=float, delimiter=','))

population = list(range(0, 5500, 500))
population[0] = 100

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for label, r in zip(folder, results):
    ax.plot(population, r, label=label)
ax.set_xlabel('Population')
ax.set_ylabel('Saturation point')
ax.legend()
plt.savefig('S-Curve_OnRate_Cumulative.png', dpi=150, format='png')
plt.show()

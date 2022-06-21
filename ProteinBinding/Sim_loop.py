import matplotlib.pyplot as plt
import numpy as np
import time
from Simulation import Simulate
from Settings import LIMIT, TOTAL_TIME, ON, OFF


population = list(range(0, 5500, 500))
population[0] = 100
limit = LIMIT

results = []
s_curve = []
start = time.time()
for pop in population:
    print(f'\tChecking on-rate: {pop}')
    _, __, stuck = Simulate(size=pop, limit=limit)
    results.append(stuck)
    s_curve.append(np.mean(stuck[-1000:]))

np.savetxt(f'Pop{population[0]}-{population[-1]}.csv',
           results,
           delimiter=',')
np.savetxt(f'S-Curve_Pop{population[0]}-{population[-1]}.csv',
           s_curve,
           delimiter=',')
end = time.time()
print(f'@@@@@  Simulation took: {end - start}s @@@@@')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for label, r in zip(population, results):
    ax.plot(r, label=f'Population: {label}')
ax.set_xlim([0, (1.1 * TOTAL_TIME)])
ax.set_ylim([0, 200])
ax.set_xlabel('Time')
ax.set_ylabel('Number of bound particles')
ax.legend()
plt.title(f'On-rate: {ON:.5f}, Off-rate: {OFF:.5f}')
plt.savefig(f'Population_On-{ON:.5f}_Off-{OFF:.5f}.png', dpi=150, format='png')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(population, s_curve)
ax.set_xlabel('Population')
ax.set_ylabel('Saturation point')
plt.title(f'On-rate: {ON:.5f}, Off-rate: {OFF:.5f}')
plt.savefig(f'S-Curve_On-{ON:.5f}_Off-{OFF:.5f}.png', dpi=150, format='png')

import matplotlib.pyplot as plt
import numpy as np
from Simulation import Simulate
from Settings import LIMIT, TOTAL_TIME, SIZE

on_rates = np.arange(0, 1.2, 0.2)
off = 0
fname = f'OffRate-{off*1000:.0f}.png'
limit = LIMIT
results = []

for on in on_rates:
    print(f'Checking on-rate: {on}')
    _, __, stuck = Simulate(on=on, off=off, limit=limit)
    results.append(stuck)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for label, r in zip(on_rates, results):
    ax.plot(r, label=f'On rate: {label:.1f}')
ax.set_xlim([0, (1.1 * TOTAL_TIME)])
ax.set_ylim([0, (1.1 * limit)])
ax.set_xlabel('Time')
ax.set_ylabel('Number of bound particles')
ax.legend()
plt.title(f'Particles: {SIZE}, Off-rate: {off:.3f}')
plt.savefig(fname, dpi=150, format='png')

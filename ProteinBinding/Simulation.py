import numpy as np
import matplotlib.pyplot as plt
from Boundary import Boundary
from Population import Population

b_y = 50
b_x = 900

'''Create boundaries'''
'''
Must be defined in top left quadrant, i.e. x must be negative, y positive
'''
boundaries = []
boundaries.append(Boundary(5, (-50, -b_y), (50, -b_y), True, 0.5, 0.25, 15))
boundaries.append(Boundary(6, (-50, b_y), (50, b_y), True, 0.5, 0.25, 15))

# boundaries.append(Boundary((-b_x, -b_y), (-b_x, b_y), False, 0.5, 0.5, 100))
# boundaries.append(Boundary((b_x, -b_y), (b_x, b_y), False, 0.5, 0.5, 100))
# boundaries.append(Boundary((-b_x, -b_y), (-50, -b_y), False, 0.5, 0.5, 100))
boundaries.append(Boundary(1, (-b_x, -b_y), (-50, -b_y), False, 0.5, 0.5, 100))
boundaries.append(Boundary(1, (-50, -b_y), (b_x, -b_y), False, 0.5, 0.5, 100))

boundaries.append(Boundary(2, (-b_x, b_y), (-50, b_y), False, 0.5, 0.5, 100))
boundaries.append(Boundary(2, (50, b_y), (b_x, b_y), False, 0.5, 0.5, 100))

boundaries.append(Boundary(3, (b_x, b_y), (b_x, -b_y), False, 0.5, 0.5, 100))
boundaries.append(Boundary(4, (-b_x, b_y), (-b_x, -b_y), False, 0.5, 0.5, 100))


'''Create a Population'''
size = 1000
x = np.random.uniform(-600, -200, size)
y = np.random.uniform(-45, 45, size)
p = Population(size, x, y, boundaries)

'''Update simulation'''
fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

total_time = 1000
filter_plot = 250
dist_before = []
dist_after = []
stuck = []
for i in range(total_time + 1):
    p.update_brownian_velocity(5)
    p.update_Poiseuille_velocity(np.array([5, 0]), b_y)
    p.update_linear_velocity(np.array([5, 0]))
    p.check_boundaries()
    p.update_position()
    p.replace_particles(old_pos=[210, None], new_pos=[[-200, -200], [-45, 45]])
    # print(i)

    dist_before.append(p.get_dist(x_plane=-80))
    dist_after.append(p.get_dist(x_plane=80))
    stuck.append(p.get_stuck())

    if i % filter_plot == 0:
        ax.clear()
        ax1.clear()
        ax2.clear()

        ax.set_xlim([-200, 200])
        ax.set_ylim([-55, 55])
        ax1.set_xlim([-200, 200])
        ax1.set_ylim([0, size/10])
        ax2.set_xlim([-200, 200])
        ax2.set_ylim([0, size/10])

        for b in boundaries:
            b.plot(ax)

        p.plot(ax, history=False)
        ax1.hist(dist_before[-1])
        ax2.hist(dist_after[-1])
        plt.title(f'Time step: {i} of {total_time}')
        plt.pause(0.01)

dist_before = np.concatenate(np.array(dist_before, dtype='object'))
dist_after = np.concatenate(np.array(dist_after, dtype='object'))
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(8, 4))
ax.hist(dist_before, color='red', bins=20, alpha=0.5, label='Before')
ax.hist(dist_after, color='green', bins=20, alpha=0.5, label='After')
ax.legend()
ax.set_xlim([-100, 100])
ax1.plot(stuck)
plt.show()

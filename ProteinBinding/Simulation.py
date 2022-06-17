import numpy as np
import matplotlib.pyplot as plt
# from Particle import Particle
from Domain import Domain
from Boundary import Boundary
from Population import Population

b_y = 50
b_x = 900
'''Create domain'''
domain = Domain((-(b_x + 2), -(b_y + 2)), ((b_x + 2), (b_y + 2)))

'''Create boundaries'''
'''
Must be defined in top left quadrant, i.e. x must be negative, y positive
'''
boundaries = []
# boundaries.append(Boundary((-50, -b_y), (50, -b_y), True, 1, 0))
# boundaries.append(Boundary((-50, b_y), (50, b_y), True, 1, 0))

# boundaries.append(Boundary((-b_x, -b_y), (-b_x, b_y), False, 0.5, 0.5))
# boundaries.append(Boundary((b_x, -b_y), (b_x, b_y), False, 0.5, 0.5))
# boundaries.append(Boundary((-b_x, -b_y), (-50, -b_y), False, 0.5, 0.5))
boundaries.append(Boundary((-b_x, -b_y), (b_x, -b_y), False, 0.5, 0.5))
boundaries.append(Boundary((-b_x, b_y), (b_x, b_y), False, 0.5, 0.5))
boundaries.append(Boundary((b_x, b_y), (b_x, -b_y), False, 0.5, 0.5))
boundaries.append(Boundary((-b_x, b_y), (-b_x, -b_y), False, 0.5, 0.5))


'''Create a Population'''
size = 100
# x = np.random.uniform(-0, -0, size)
# y = np.random.uniform(45, 45, size)
x = np.random.uniform(-200, -150, size)
y = np.random.uniform(-45, 45, size)
# print(f'Particle location: ({x}, {y})')
p = Population(size, x, y, boundaries, domain)

'''Update simulation'''
fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

total_time = 50
filter_plot = 1
dist_before = []
dist_after = []
stuck = []
for i in range(total_time + 1):
    # print('History:')
    # print(f'\tX: {p.hist_x[-5:]}')
    # print(f'\tY: {p.hist_y[-5:]}')
    # print(f'History Length:\t{len(p.hist_x)}')
    p.update_brownian_velocity(5)
    # p.update_Poiseuille_velocity(np.array([5, 0]), b_y)
    p.update_linear_velocity(np.array([5, 0]))
    p.check_boundaries()
    # p.update_position(ax=ax1)
    print(i)

    dist_before.append(p.get_dist(x_plane=-80))
    dist_after.append(p.get_dist(x_plane=80))
    stuck.append(p.get_stuck())

    if i % filter_plot == 0:
        ax.clear()
        ax1.clear()
        ax2.clear()

        ax.set_xlim([-200, 200])
        ax.set_ylim([-55, 55])
        # ax1.set_xlim([-200, 200])
        # ax1.set_ylim([0, size/10])
        # ax2.set_xlim([-200, 200])
        # ax2.set_ylim([0, size/10])

        # domain.plot(ax)
        for b in boundaries:
            b.plot(ax)
        p.update_position(ax=ax1)
        p.plot(ax, history=True)
        # ax1.hist(dist_before[-1])
        # ax2.hist(dist_after[-1])
        plt.title(f'Time step: {i} of {total_time}')
        plt.pause(0.01)

# dist_before = np.concatenate(np.array(dist_before, dtype='object'))
# dist_after = np.concatenate(np.array(dist_after, dtype='object'))
# fig, (ax, ax1) = plt.subplots(1, 2, figsize=(8, 4))
# ax.hist(dist_before, color='red', bins=20, alpha=0.5, label='Before')
# ax.hist(dist_after, color='green', bins=20, alpha=0.5, label='After')
# ax.legend()
# ax.set_xlim([-100, 100])
# ax1.plot(stuck)
plt.show()

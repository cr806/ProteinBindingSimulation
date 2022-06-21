import time
import numpy as np
import matplotlib.pyplot as plt
from Boundary import Boundary
from Population import Population
from Domain import Domain
from Settings import SIZE, TOTAL_TIME, LIVE_PLOT, PLOT_EVERY
from Settings import HIST_BEFORE, HIST_AFTER, PLOT_HISTORY
from Settings import ON, OFF, LIMIT


def Simulate(on=ON, off=OFF, limit=LIMIT, size=SIZE):
    '''
    CREATE BOUNDARIES
    '''
    b_y = 50
    b_x = 700
    boundaries = []
    boundaries.append(Boundary(5, (-50, -b_y), (50, -b_y),
                               True, on, off, limit))
    # boundaries.append(Boundary(6, (-50, b_y), (50, b_y),
    #                            True, 0.5, 0.25, 15))

    # boundaries.append(Boundary((-b_x, -b_y), (-b_x, b_y),
    #                            False, 0.5, 0.5, 100))
    # boundaries.append(Boundary((b_x, -b_y), (b_x, b_y),
    #                            False, 0.5, 0.5, 100))
    # boundaries.append(Boundary((-b_x, -b_y), (-50, -b_y),
    #                            False, 0.5, 0.5, 100))
    boundaries.append(Boundary(1, (-b_x, -b_y), (-50, -b_y),
                               False, 0.5, 0.5, 100))
    boundaries.append(Boundary(1, (-50, -b_y), (b_x, -b_y),
                               False, 0.5, 0.5, 100))

    boundaries.append(Boundary(2, (-b_x, b_y), (b_x, b_y),
                               False, 0.5, 0.5, 100))
    # boundaries.append(Boundary(2, (50, b_y), (b_x, b_y),
    #                            False, 0.5, 0.5, 100))

    boundaries.append(Boundary(3, (b_x, b_y), (b_x, -b_y),
                               False, 0.5, 0.5, 100))
    boundaries.append(Boundary(4, (-b_x, b_y), (-b_x, -b_y),
                               False, 0.5, 0.5, 100))

    '''CREATE SIMULATION DOMAIN'''
    d = Domain([-b_x, -b_y], [b_x, b_y])

    '''CREATE POPULATION'''
    x = np.random.uniform(-600, -200, size)
    y = np.random.uniform(-45, 45, size)
    p = Population(size, x, y, boundaries)

    '''UPDATE SIMULATION'''
    if LIVE_PLOT:
        fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

    dist_before = []
    dist_after = []
    stuck = []
    start = time.time()
    for i in range(TOTAL_TIME + 1):
        p.update_brownian_velocity(1)
        p.update_Poiseuille_velocity(np.array([6, 0]), b_y)
        # p.update_linear_velocity(np.array([5, 0]))
        p.check_boundaries()
        p.replace_particles(old_pos=[210, None],
                            new_pos=[[-600, -200], [-45, 45]])
        p.update_position()
        d.check_extent(p.x, p.y)
        # print(f'Simulation interation: {i} / {TOTAL_TIME}')

        dist_before.append(p.get_dist(x_plane=HIST_BEFORE))
        dist_after.append(p.get_dist(x_plane=HIST_AFTER))
        stuck.append(p.get_stuck())

        if LIVE_PLOT and (i % PLOT_EVERY == 0):
            ax.clear()
            ax1.clear()
            ax2.clear()

            ax.set_xlim([-200, 200])
            ax.set_ylim([-55, 55])
            ax1.set_xlim([-200, 200])
            ax1.set_ylim([0, SIZE/10])
            ax2.set_xlim([-200, 200])
            ax2.set_ylim([0, SIZE/10])

            for b in boundaries:
                b.plot(ax)

            p.plot(ax, history=PLOT_HISTORY)
            ax1.hist(dist_before[-1])
            ax2.hist(dist_after[-1])
            plt.title(f'Time step: {i} of {TOTAL_TIME}')
            plt.pause(0.01)

    end = time.time()
    print(f'Simulation time: {end - start} s')

    dist_before = np.concatenate(np.array(dist_before, dtype='object'))
    dist_after = np.concatenate(np.array(dist_after, dtype='object'))

    return dist_before, dist_after, stuck


if __name__ == '__main__':
    limit = LIMIT
    dist_before, dist_after, stuck = Simulate(on=0, limit=limit)

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    ax.hist(dist_before, color='red', bins=20, alpha=0.5, label='Before')
    ax.hist(dist_after, color='green', bins=20, alpha=0.5, label='After')
    ax.legend()
    ax.set_xlim([-100, 100])
    ax1.plot(stuck)
    ax1.set_xlim([0, (1.1 * TOTAL_TIME)])
    ax1.set_ylim([0, (1.1 * limit)])
    plt.show()

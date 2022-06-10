import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.collections as coll
from matplotlib.animation import PillowWriter
from itertools import combinations

'''Program general parameters'''
n_particles = 400
total_time = 1000
particle_speed = 500
particle_r = 0.015

plot_initial = False
live_plot = False
animate_save = True
animation_name = 'ani4.gif'

'''
Define number of particles and get random positions (between 0 and 1)
for each particle
'''
pos = np.random.random((2, n_particles))
# Color particles the start on either side
ixr = pos[0] > 0.5  # right
ixl = pos[0] <= 0.5  # left

'''Give IDs to each particle (this will be useful later)'''
ids = np.arange(n_particles)

'''Plot initial configuration of particles'''
if plot_initial:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(pos[0][ixr], pos[1][ixr], color='r', s=6)
    ax.scatter(pos[0][ixl], pos[1][ixl], color='b', s=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()

'''
Set initial velocities of particles. RHS move to the left and vice versa.
'''
v = np.random.uniform(-particle_speed, particle_speed, (2, n_particles))
# v = np.zeros((2, n_particles))
# v[0][ixr] = -particle_speed
# v[0][ixl] = particle_speed

'''
Distance between all pairs
Need to calculate the distance between all pairs of particles. If the distance
is less than 2 times the radius of each particle, they collide. If there are
n_particles, there are (n * (n -1) / 2) pairs (combinatorics). To make life
easier, calculate pairs of particle IDs.
'''
ids_pairs = np.asarray(list(combinations(ids, 2)))
# x_pairs = np.asarray(list(combinations(pos[0], 2)))  # pairs of x-positions
# y_pairs = np.asarray(list(combinations(pos[1], 2)))
# dx_pairs = np.diff(x_pairs, axis=1).ravel()  # difference between x-pairs
# dy_pairs = np.diff(y_pairs, axis=1).ravel()
# d_pairs = np.sqrt(dx_pairs**2 + dy_pairs**2)  # Euclidean distance

'''
Velocities of particles after a collision
Each iteration, evaluate d_pairs, and if any of the distances between
particles is less than 2r, then a collision occurs.
The final velocity of each of the two paticles, in an elastic collision,
v1_new = v1 -( ((v1 - v2) dot (r1 - r2)) / (mag(r1 - r2))**2) * (r1 - r2)
v2_new = v2 -( ((v2 - v1) dot (r2 - r1)) / (mag(r1 - r2))**2) * (r2 - r1)
'''
# ids_pairs_collide = ids_pairs[d_pairs < (2 * particle_r)]
# v1 = v[:, ids_pairs_collide[:, 0]]  # vel of particle 1 in collision pairs
# v2 = v[:, ids_pairs_collide[:, 1]]
# pos1 = pos[:, ids_pairs_collide[:, 0]]  # pos of particle 1 in collision pairs
# pos2 = pos[:, ids_pairs_collide[:, 1]]

# v1new = v1 - (np.diag((v1 - v2).T @ (pos1 - pos2)) /
#               np.sum((pos1 - pos2)**2, axis=0) * (pos1 - pos2))
# v2new = v2 - (np.diag((v2 - v1).T @ (pos2 - pos1)) /
#               np.sum((pos2 - pos1)**2, axis=0) * (pos2 - pos1))


'''
Functions for the above steps
'''


def get_delta_pairs(x):
    '''
    Return difference array between two particle locations (in one axis)
    '''
    x_pairs = np.asarray(list(combinations(x, 2)))
    return np.diff(x_pairs, axis=1).ravel()


def get_deltad_pairs(pos):
    '''
    Calculate Euclidean distance array between two particles differences
    '''
    return np.sqrt(get_delta_pairs(pos[0])**2 + get_delta_pairs(pos[1])**2)


def compute_new_v(p_speed, v1, v2, pos1, pos2):
    '''
    Calculate new velocites for colliding particles
    '''
    v1new = v1 - (np.diag((v1 - v2).T @ (pos1 - pos2)) /
                  np.sum((pos1 - pos2)**2, axis=0) * (pos1 - pos2))
    v2new = v2 - (np.diag((v2 - v1).T @ (pos2 - pos1)) /
                  np.sum((pos2 - pos1)**2, axis=0) * (pos2 - pos1))
    return v1new, v2new


def compute_mag_v(v):
    return np.linalg.norm(v, axis=0)


def motion(p_speed, pos, v, id_pairs, time_s, dt, d_cutoff):
    pos_s = np.zeros((time_s, pos.shape[0], pos.shape[1]))
    vs = np.zeros((time_s, v.shape[0], v.shape[1]))
    # Initial State
    pos_s[0] = pos.copy()
    vs[0] = v.copy()
    for i in range(1, time_s):
        ic = id_pairs[get_deltad_pairs(pos) < (d_cutoff * 2)]  # idx colliders
        v[:, ic[:, 0]], v[:, ic[:, 1]] = compute_new_v(p_speed,
                                                       v[:, ic[:, 0]],
                                                       v[:, ic[:, 1]],
                                                       pos[:, ic[:, 0]],
                                                       pos[:, ic[:, 1]])

        v[0, pos[0] > 1 - d_cutoff] = -np.abs(v[0, pos[0] > 1 - d_cutoff])
        v[0, pos[0] < 0 + d_cutoff] = np.abs(v[0, pos[0] < 0 + d_cutoff])
        v[1, pos[1] > 1 - d_cutoff] = -np.abs(v[1, pos[1] > 1 - d_cutoff])
        v[1, pos[1] < 0 + d_cutoff] = np.abs(v[1, pos[1] < 0 + d_cutoff])

        pos = pos + (v * dt)
        pos_s[i] = pos.copy()
        vs[i] = v.copy()
        print(f'Timestep: {i}')
    return pos_s, vs


pos_s, vs = motion(particle_speed, pos, v, ids_pairs,
                   time_s=total_time,
                   dt=0.000008,
                   d_cutoff=particle_r)

if live_plot:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for i in range(total_time):
        ax.clear()
        xred = pos_s[i][0][ixr]
        yred = pos_s[i][1][ixr]

        xblue = pos_s[i][0][ixl]
        yblue = pos_s[i][1][ixl]

        circles_red = [plt.Circle((xi, yi), radius=particle_r, linewidth=0)
                       for xi, yi in zip(xred, yred)]

        circles_blue = [plt.Circle((xi, yi), radius=particle_r, linewidth=0)
                        for xi, yi in zip(xblue, yblue)]

        cred = coll.PatchCollection(circles_red, facecolors='red')
        cblue = coll.PatchCollection(circles_blue, facecolors='blue')

        ax.add_collection(cred)
        ax.add_collection(cblue)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.pause(0.001)
    plt.show()


def animate(i):
    ax.clear()
    xred = pos_s[i][0][ixr]
    yred = pos_s[i][1][ixr]

    xblue = pos_s[i][0][ixl]
    yblue = pos_s[i][1][ixl]

    circles_red = [plt.Circle((xi, yi), radius=particle_r, linewidth=0)
                   for xi, yi in zip(xred, yred)]

    circles_blue = [plt.Circle((xi, yi), radius=particle_r, linewidth=0)
                    for xi, yi in zip(xblue, yblue)]

    cred = coll.PatchCollection(circles_red, facecolors='red')
    cblue = coll.PatchCollection(circles_blue, facecolors='blue')

    ax.add_collection(cred)
    ax.add_collection(cblue)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


if animate_save:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=50)
    ani.save(animation_name, writer='pillow', fps=30, dpi=100)

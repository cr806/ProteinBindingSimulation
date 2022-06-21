import numpy as np
import HelperFunc as hf
from Settings import SIZE

# import matplotlib.pyplot as plt


class Population:
    def __init__(self, size, initial_x, initial_y, boundaries):
        '''
        Defines attributes of the class: size, x- and y-position, and
        boundaries that must be considered
        '''
        self.size = size
        self.active = np.full(size, -1, dtype=int)
        self.boundaries = boundaries
        self.x = initial_x
        self.y = initial_y
        self.vx = np.full(size, 0, dtype=float)
        self.vy = np.full(size, 0, dtype=float)
        self.hist_x = []
        self.hist_y = []
        self.hist_x.append(np.copy(self.x))
        self.hist_y.append(np.copy(self.y))

    def get_random_velocity(self, factor=1):
        '''
        Assign a random velocity to a population
        '''
        vx = np.random.uniform(-1, 1, self.size)
        vy = np.random.uniform(-1, 1, self.size)
        vx, vy = hf.normalise_vector(vx, vy)
        return (factor * vx), (factor * vy)

    def update_brownian_velocity(self, factor):
        vel_x, vel_y = self.get_random_velocity(factor=factor)
        self.vx = self.vx + vel_x
        self.vy = self.vy + vel_y

    def update_Poiseuille_velocity(self, factor, h):
        self.vx = self.vx + factor[0] * ((h**2 - self.y**2) / h**2)
        self.vy = self.vy + factor[1] * ((h**2 - self.x**2) / h**2)

    def update_linear_velocity(self, vel):
        vel_x = np.full(self.size, vel[0], dtype=float)
        vel_y = np.full(self.size, vel[1], dtype=float)
        self.vx = self.vx + vel_x
        self.vy = self.vy + vel_y

    def update_position(self, ax=None):
        update = self.active == -1
        self.x = self.x + (update * self.vx)
        self.y = self.y + (update * self.vy)
        self.hist_x.append(np.copy(self.x))
        self.hist_y.append(np.copy(self.y))
        if (ax is not None):
            ax.plot(update, 'o')
        self.vx = np.full(self.size, 0, dtype=float)
        self.vy = np.full(self.size, 0, dtype=float)

    def check_boundaries(self):
        pos = hf.combineVectors(self.x, self.y)
        vel = hf.combineVectors(self.vx, self.vy)
        f_pos = pos + vel
        to_check = np.full(self.size, True, dtype=bool)

        for b in self.boundaries:
            already_stuck = self.active == b.ID
            no_already_stuck = np.count_nonzero(already_stuck)
            boundary_full = False
            if b.sticky and (no_already_stuck >= b.limit):
                boundary_full = True

            free_particles = self.active == -1
            direction = self.check_particle_direction(b, pos, f_pos)
            reach = self.check_particle_reaches(b, pos, f_pos)
            int_pos = self.intersect_positions(b, pos)
            ref_pos = self.reflected_positions(b, f_pos)
            overlap = self.check_overlaps_boundary(b, int_pos)
            dir_reach = direction * reach * overlap

            to_update = dir_reach * to_check * free_particles

            if b.sticky and not boundary_full:
                chance_to_stick = np.random.uniform(0, 1, self.size)
                chance_to_stick = chance_to_stick < b.on
                update = to_update * chance_to_stick

                to_stick = b.limit - no_already_stuck
                idxs = update.nonzero()[0]
                if idxs.size > to_stick:
                    update[idxs[to_stick]:] = False
                    self.x[idxs[to_stick:]] = ref_pos[:, 0][idxs[to_stick:]]
                    self.y[idxs[to_stick:]] = ref_pos[:, 1][idxs[to_stick:]]
                    self.vx[idxs[to_stick:]] = 0
                    self.vy[idxs[to_stick:]] = 0
                self.x[update] = int_pos[:, 0][update]
                self.y[update] = int_pos[:, 1][update]
                self.vx[update] = 0
                self.vy[update] = 0
                self.active[update] = b.ID
                to_check[dir_reach] = False
            else:
                self.x[to_update] = ref_pos[:, 0][to_update]
                self.y[to_update] = ref_pos[:, 1][to_update]
                self.vx[to_update] = 0
                self.vy[to_update] = 0
                to_check[dir_reach] = False

            for b in self.boundaries:
                to_update = self.active == b.ID
                unstick = np.random.uniform(0, 1, self.size)
                unstick = unstick < b.off
                update = to_update * unstick
                self.active[update] = -1

        pos[to_check] = pos[to_check] + (vel[to_check] / 2)
        self.hist_x.append(np.copy(pos[:, 0]))
        self.hist_y.append(np.copy(pos[:, 1]))

    def check_particle_direction(self, b, pos, f_pos):
        horizontal = b.direction[0]
        b_start = np.full([self.size, 2], b.start, dtype=float)
        if horizontal:
            p_dist = f_pos[:, 1] - pos[:, 1]
            b_dist = b_start[:, 1] - pos[:, 1]
            return p_dist * b_dist > 0
        else:
            p_dist = f_pos[:, 0] - pos[:, 0]
            b_dist = b_start[:, 0] - pos[:, 0]
            return p_dist * b_dist > 0

    def check_particle_reaches(self, b, pos, f_pos):
        horizontal = b.direction[0]
        b_start = np.full([self.size, 2], b.start, dtype=float)
        if horizontal:
            p_dist = f_pos[:, 1] - pos[:, 1]
            b_dist = b_start[:, 1] - pos[:, 1]
            return np.abs(p_dist) >= np.abs(b_dist)
        else:
            p_dist = f_pos[:, 0] - pos[:, 0]
            b_dist = b_start[:, 0] - pos[:, 0]
            return np.abs(p_dist) >= np.abs(b_dist)

    def intersect_positions(self, b, pos):
        horizontal = b.direction[0]
        b_start = np.full([self.size, 2], np.abs(b.start), dtype=float)
        b_sign = np.full([self.size, 2], np.sign(b.start), dtype=float)
        if horizontal:
            # Ensure particles stay within domain, never on boundary so that
            # when they 'unstick' from a boundary they are not able to leave
            # simulation domain - direction check is invalid for zero
            int_pos = hf.combineVectors(pos[:, 0],
                                        b_sign[:, 1] * (b_start[:, 1] - 0.1))
        else:
            int_pos = hf.combineVectors(b_sign[:, 0] * (b_start[:, 0] - 0.1),
                                        pos[:, 1])
        return int_pos

    def reflected_positions(self, b, f_pos):
        horizontal = b.direction[0]
        b_start = np.full([self.size, 2], b.start, dtype=float)
        if horizontal:
            ref_pt_x = f_pos[:, 0]
            ref_pt_y = b_start[:, 1] - (f_pos[:, 1] - b_start[:, 1])
        else:
            ref_pt_x = b_start[:, 0] - (f_pos[:, 0] - b_start[:, 0])
            ref_pt_y = f_pos[:, 1]
        return hf.combineVectors(ref_pt_x, ref_pt_y)

    def replace_particles(self, old_pos, new_pos):
        # Reposition particles that have reached certain position
        if old_pos[0]:
            to_reposition = self.x >= old_pos[0]
        else:
            to_reposition = self.y >= old_pos[1]
        num = np.count_nonzero(to_reposition)
        self.reset_particle_position(to_reposition, new_pos, num)

        # Compensate for particles trapped on boundaries
        to_add = SIZE - np.count_nonzero(self.active == -1)
        if to_add > 0:
            self.create_new_particles(new_pos, to_add)

    def reset_particle_position(self, to_reposition, new_pos, num):
        self.x[to_reposition] = np.random.uniform(new_pos[0][0],
                                                  new_pos[0][1],
                                                  num)
        self.y[to_reposition] = np.random.uniform(new_pos[1][0],
                                                  new_pos[1][1],
                                                  num)

    def create_new_particles(self, new_pos, to_add):
        x_ = np.random.uniform(new_pos[0][0], new_pos[0][1], to_add)
        y_ = np.random.uniform(new_pos[1][0], new_pos[1][1], to_add)
        vx_ = np.full(to_add, 0, dtype=float)
        vy_ = np.full(to_add, 0, dtype=float)
        active_ = np.full(to_add, -1, dtype=int)
        self.x = np.concatenate((self.x, x_))
        self.vx = np.concatenate((self.y, vx_))
        self.y = np.concatenate((self.y, y_))
        self.vy = np.concatenate((self.vy, vy_))
        self.active = np.concatenate((self.active, active_))
        self.update_size()

    def update_size(self):
        self.size = self.x.size

    def check_overlaps_boundary(self, b, int_pos):
        b_start = np.full([self.size, 2], b.start, dtype=float)
        b_end = np.full([self.size, 2], b.end, dtype=float)
        horizontal = b.direction[0]
        if horizontal:
            overlap1 = ((int_pos[:, 0] >= b_start[:, 0]) *
                        (int_pos[:, 0] <= b_end[:, 0]))
            overlap2 = ((int_pos[:, 0] <= b_start[:, 0]) *
                        (int_pos[:, 0] >= b_end[:, 0]))
            return (overlap1 + overlap2)
        else:
            overlap1 = ((int_pos[:, 1] >= b_start[:, 1]) *
                        (int_pos[:, 1] <= b_end[:, 1]))
            overlap2 = ((int_pos[:, 1] <= b_start[:, 1]) *
                        (int_pos[:, 1] >= b_end[:, 1]))
            return (overlap1 + overlap2)

    # def find_intersection_point(self, bl, bu):
    #     '''
    #     Calculate the point at which the particle would intersect
    #     with the boundary if the boundary was infinite in length
    #     '''
    #     # Check for horizontal or vertical trajectory or boundary
    #     p_vert = self.vx == 0
    #     p_hor = self.vy == 0
    #     p_angle = ~(np.logical_or(p_vert, p_hor))
    #     b_vert, b_hor = (bl[0] - bu[0]) == 0
    #     b_angle = not(b_vert or b_hor)

    #     # print(f'Particle: {p_vert}, {p_hor}, {p_angle}')
    #     # print(f'Boundary: {b_vert}, {b_hor}, {b_angle}')

    #     # Return intersection if particle trajectory is vertical
    #     # or horizontal
    #     mp, cp = self.get_line_equ(self.x, self.y,
    #                                self.x + self.vx, self.y + self.vy)
    #     mp[mp == 0] = 0.00001  # Used to avoid divide by zero error

    #     bx = bl[0, 0]
    #     by = bl[0, 1]
    #     if b_vert:
    #         intersection_x = bl[:, 0]
    #         intersection_y = ((p_vert * 100000) +
    #                           (p_hor * self.y) +
    #                           (p_angle * ((mp * bx) + cp)))
    #     if b_hor:
    #         intersection_x = ((p_vert * self.x) +
    #                           (p_hor * 100000) +
    #                           (p_angle * ((by - cp) / mp)))
    #         intersection_y = bl[:, 1]
    #     if b_angle:
    #         m, c = self.get_line_equ(bl[:, 0], bl[:, 1], bu[:, 0], bu[:, 1])
    #         m[m == 0] = 0.00001  # Used to avoid divide by zero error
    #         intersection_x = ((p_vert * self.x) +
    #                           (p_hor * (self.y - c) / m) +
    #                           (p_angle * ((c - cp) / (mp - m))))
    #         intersection_y = ((p_vert * (m * self.x) + c) +
    #                           (p_hor * self.y) +
    #                           (p_angle * (((cp * m) - (c * mp)) / (mp - m))))

    #     return hf.combineVectors(intersection_x, intersection_y)

    # def get_line_equ(self, a_x, a_y, b_x, b_y):
    #     '''
    #     Find equations of particle tradjectory and boundary
    #     i.e y = mx + c
    #     '''
    #     dy = b_y - a_y
    #     dx = b_x - a_x
    #     dx[dx == 0] = 0.00001  # avoid divide-by-zero
    #     m = dy / dx
    #     c = b_y - (m * b_x)

    #     return m, c

    def get_dist(self, x_plane=False, y_plane=False):
        if x_plane:
            gt = self.x > (x_plane - 10)
            lt = self.x < (x_plane + 10)
            return self.y[np.logical_and(gt, lt)]
        if y_plane:
            gt = self.y > (y_plane - 10)
            lt = self.y < (y_plane + 10)
            return self.x[np.logical_and(gt, lt)]

    def get_stuck(self):
        update = self.active == -1
        return np.count_nonzero(~update)

    def print_data(self):
        print('\tPos\tVel')
        print(f'\t({self.x}, {self.y})\t({self.vx}, {self.vy})')

    def plot(self, ax, history=False):
        update = self.active == -1
        ax.plot(self.x[update],
                self.y[update], '*')
        ax.plot(self.x[~update],
                self.y[~update],
                'go', alpha=0.5)

        if history:
            h_x = np.array(self.hist_x)
            h_y = np.array(self.hist_y)
            drawlines = []
            for data_x, data_y in zip(h_x.T, h_y.T):
                drawlines.append(data_x)
                drawlines.append(data_y)
            ax.plot(*drawlines,
                    color='blue',
                    linestyle='dotted',
                    marker='o',
                    markersize='0.5',
                    linewidth='0.5')

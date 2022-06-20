import numpy as np
import HelperFunc as hf
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
        self.hist_x.append(self.x.copy())
        self.hist_y.append(self.y.copy())

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
        self.hist_x.append(self.x.copy())
        self.hist_y.append(self.y.copy())
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
            update = self.active == b.ID
            unstick = np.random.uniform(0, 1, self.size)
            unstick = unstick <= b.off
            update = update * unstick
            self.active[update] = -1

        for b in self.boundaries:
            free_particles = self.active == -1
            direction = self.check_particle_direction(b, pos, f_pos)
            reach = self.check_particle_reaches(b, pos, f_pos)
            int_pos = self.intersect_positions(b, pos)
            ref_pos = self.reflected_positions(b, f_pos)
            overlap = self.check_overlaps_boundary(b, int_pos)
            dir_reach = direction * reach * overlap

            to_update = dir_reach * to_check * free_particles

            if b.sticky:
                self.x[to_update] = int_pos[:, 0][to_update]
                self.y[to_update] = int_pos[:, 1][to_update]
                self.vx[to_update] = 0
                self.vy[to_update] = 0
                self.active[to_update] = b.ID
                to_check[dir_reach] = False
            else:
                self.x[to_update] = ref_pos[:, 0][to_update]
                self.y[to_update] = ref_pos[:, 1][to_update]
                self.vx[to_update] = 0
                self.vy[to_update] = 0
                to_check[dir_reach] = False

        pos[to_check] = pos[to_check] + (vel[to_check] / 2)
        self.hist_x.append(pos[:, 0].copy())
        self.hist_y.append(pos[:, 1].copy())

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
        to_replace = np.full(self.size, False, dtype=bool)
        if old_pos[0]:
            to_replace[self.x >= old_pos[0]] = True
        else:
            to_replace[self.x >= old_pos[0]] = True

        num = np.count_nonzero(to_replace)
        self.x[to_replace] = np.random.uniform(new_pos[0][0],
                                               new_pos[0][1],
                                               num)
        self.y[to_replace] = np.random.uniform(new_pos[1][0],
                                               new_pos[1][1],
                                               num)

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

    # def check_boundary(self):
    #     '''
    #     Check whether particle has interacted with a boundary
    #     '''
    #     print('Check boundary')
    #     self.print_data()
    #     plot = False
    #     pos = hf.combineVectors(self.x, self.y)
    #     vel = hf.combineVectors(self.vx, self.vy)
    #     f_pos = pos + vel
    #     for idx, b in enumerate(self.boundaries):
    #         print(f'1. Boundary {idx}')
    #         self.print_data()
    #         bl = np.full([self.size, 2], b.start, dtype=float)
    #         bu = np.full([self.size, 2], b.end, dtype=float)
    #         b_dir = np.full([self.size, 2], b.direction, dtype=bool)
    #         b_normal = np.full([self.size, 2], b.normal, dtype=float)
    #         overlap = np.full(self.size, False, dtype=bool)
    #         print(f'2. Boundary {idx}')
    #         self.print_data()

    #         # Calculate distance moved by particle and distance to
    #         # boundary intersection point

    #         # new_vel = hf.reflectVector(b_normal, vel)
    #         new_vel_x = self.vx - 2 * (b_dir[:, 1] * self.vx)
    #         new_vel_y = self.vy - 2 * (b_dir[:, 0] * self.vy)
    #         new_vel = hf.combineVectors(new_vel_x, new_vel_y)
    #         print(f'3. Boundary {idx}')
    #         print(f'\tNew V:\t{new_vel}')
    #         self.print_data()

    #         int_xy = self.find_intersection_point(bl, bu)
    #         print(f'4. Boundary {idx}')
    #         print(f'\tNew V:\t{new_vel}')
    #         print(f'\tIntersection;\t{int_xy}')
    #         self.print_data()

    #         if plot:
    #             fig, ax = plt.subplots(figsize=(5, 5))
    #             for i in range(self.size):
    #                 ax.plot([0, vel[i, 0]],
    #                         [0, vel[i, 1]],
    #                         'o-',
    #                         markersize=20,
    #                         label=f'Inc. {i}')
    #                 ax.plot([0, new_vel[i, 0]],
    #                         [0, new_vel[i, 1]],
    #                         'o-',
    #                         markersize=10,
    #                         label=f'Ref. {i}')
    #                 ax.plot([0, b_normal[0, 0]],
    #                         [0, b_normal[0, 1]],
    #                         'o-',
    #                         label=f'B. {b_normal[0]}')
    #             ax.legend()
    #             plot = False

    #         # Check if particles can reach boundary
    #         # particle_dist = np.linalg.norm(vel, axis=1)
    #         # dist_to_boundary = np.linalg.norm((int_xy - pos), axis=1)

    #         # direction = np.diagonal(np.dot((int_xy - pos), vel.T)) > 1

    #         # direction = (np.diag((int_xy - pos).T @ vel) /
    #         #             np.sum((int_xy - pos)**2, axis=0) > 1)

    #         # reach = dist_to_boundary <= particle_dist
    #         # dir_reach = np.logical_and(direction, reach)

    #         reach_horizontal = (b_dir[:, 0] *
    #                             (np.abs(f_pos[:, 1]) > np.abs(bl[:, 1])))
    #         reach_vertical = (b_dir[:, 1] *
    #                           (np.abs(f_pos[:, 0]) > np.abs(bl[:, 0])))
    #         # print(reach_horizontal + reach_vertical)
    #         dir_reach = reach_horizontal + reach_vertical
    #         print(f'5. Boundary {idx}')
    #         print(f'\tNew V:\t{new_vel}')
    #         print(f'\tIntersection;\t{int_xy}')
    #         print(f'\tDir-Reach:\t{dir_reach}')
    #         self.print_data()

    #         # overlap[dir_reach] = self.check_boundary_overlap(bl[dir_reach],
    #         #                                                  bu[dir_reach],
    #         #                                              int_xy[dir_reach])
    #         overlap[dir_reach] = self.check_boundary_overlap(bl[dir_reach],
    #                                                          bu[dir_reach],
    #                                                          int_xy[dir_reach])

    #         self.hist_x.append(self.x)
    #         self.hist_y.append(self.y)
    #         self.x[overlap] = int_xy[:, 0][overlap]
    #         self.y[overlap] = int_xy[:, 1][overlap]
    #         self.vx[overlap] = new_vel[:, 0][overlap]
    #         self.vy[overlap] = new_vel[:, 1][overlap]
    #         # self.hist_x.append(self.x)
    #         # self.hist_y.append(self.y)
    #         print(f'6. Boundary {idx}')
    #         print(f'\tNew V:\t{new_vel}')
    #         print(f'\tIntersection;\t{int_xy}')
    #         print(f'\tDir-Reach:\t{dir_reach}')
    #         print(f'\tOverlap:\t{overlap}')
    #         self.print_data()

    #         if b.sticky:
    #             to_stick = np.random.uniform(0, 1, self.size) < b.on
    #             stuck = np.logical_and(overlap, to_stick)
    #             to_unstick = np.random.uniform(0, 1, self.size) < b.off
    #             update = self.active == -1
    #             unstuck = np.logical_and(~update, to_unstick)
    #             # self.x[unstuck] = (self.x[unstuck] +
    #             #                    (update[unstuck] *
    #                                   self.vx[unstuck]))
    #             # self.y[unstuck] = (self.y[unstuck] +
    #             #                    (update[unstuck] *
    #                                   self.vy[unstuck]))
    #             print(f'7. Boundary {idx}')
    #             print(f'\tNew V:\t{new_vel}')
    #             print(f'\tIntersection;\t{int_xy}')
    #             print(f'\tDir-Reach:\t{dir_reach}')
    #             print(f'\tOverlap:\t{overlap}')
    #             print(f'\tStuck:\t{stuck}')
    #             print(f'\tUn-stuck:\t{unstuck}')
    #             print(f'\tActive:\t{self.active}')
    #             self.print_data()
    #             self.active[unstuck] = -1
    #             self.active[stuck] = b.ID
    #             print(f'7. Boundary {idx}')
    #             print(f'\toverlap:\t{overlap}')
    #             print(f'\tNew V:\t{new_vel}')
    #             print(f'\tIntersection;\t{int_xy}')
    #             print(f'\tDir-Reach:\t{dir_reach}')
    #             print(f'\tOverlap:\t{overlap}')
    #             print(f'\tStuck:\t{stuck}')
    #             print(f'\tUn-stuck:\t{unstuck}')
    #             print(f'\tActive:\t{self.active}')

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

    # def check_boundary_overlap(self, bl, bu, int_xy):
    #     overlap_x = (((int_xy[:, 0] <= bu[:, 0]) *
    #                   (int_xy[:, 0] >= bl[:, 0])) +
    #                  ((bu[:, 0] <= int_xy[:, 0]) *
    #                   (bl[:, 0] >= int_xy[:, 0])))
    #     overlap_y = (((int_xy[:, 1] <= bu[:, 1]) *
    #                   (int_xy[:, 1] >= bl[:, 1])) +
    #                  ((bu[:, 1] <= int_xy[:, 1]) *
    #                   (bl[:, 1] >= int_xy[:, 1])))
    #     return (overlap_x * overlap_y)

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

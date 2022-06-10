import numpy as np
import HelperFunc as hf
import matplotlib.pyplot as plt


class Population:
    def __init__(self, size, initial_x, initial_y, boundaries, domain):
        '''
        Defines attributes of the class: size, x- and y-position, and
        boundaries that must be considered
        '''
        self.size = size
        self.active = np.full(size, True, dtype=bool)
        self.boundaries = boundaries
        self.domain = domain
        self.x = initial_x
        self.y = initial_y
        self.vx = np.full(size, 0, dtype=float)
        self.vy = np.full(size, 0, dtype=float)
        self.hist_x = []
        self.hist_y = []

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

    def update_flow_velocity(self, vel):
        vel_x = np.full(self.size, vel[0], dtype=float)
        vel_y = np.full(self.size, vel[1], dtype=float)
        self.vx = self.vx + vel_x
        self.vy = self.vy + vel_y

    def update_position(self):
        self.hist_x.append(self.x)
        self.hist_y.append(self.y)
        self.x = self.x + (self.active * self.vx)
        self.y = self.y + (self.active * self.vy)
        # self.x, self.y = self.domain.check_extent(self.x, self.y)
        self.vx = np.full(self.size, 0, dtype=float)
        self.vy = np.full(self.size, 0, dtype=float)

    def check_boundary(self):
        '''
        Check whether particle has interacted with a boundary
        '''
        plot = False
        for b in self.boundaries:
            bl = np.full([self.size, 2], b.lower, dtype=float)
            bu = np.full([self.size, 2], b.upper, dtype=float)
            b_normal = np.full([self.size, 2], b.normal, dtype=float)
            overlap = np.full(self.size, False, dtype=bool)

            # Calculate distance moved by particle and distance to
            # boundary intersection point
            pos = hf.combineVectors(self.x, self.y)
            vel = hf.combineVectors(self.vx, self.vy)
            new_vel = hf.reflectVector(b_normal, vel)
            int_xy = self.find_intersection_point(bl, bu)

            if plot:
                fig, ax = plt.subplots(figsize=(5, 5))
                for i in range(self.size):
                    ax.plot([0, vel[i, 0]],
                            [0, vel[i, 1]],
                            'o-',
                            markersize=20,
                            label=f'Inc. {i}')
                    ax.plot([0, new_vel[i, 0]],
                            [0, new_vel[i, 1]],
                            'o-',
                            markersize=10,
                            label=f'Ref. {i}')
                    ax.plot([0, b_normal[0, 0]],
                            [0, b_normal[0, 1]],
                            'o-',
                            label=f'B. {b_normal[0]}')
                ax.legend()
                plot = False

            # Check if particles can reach boundary
            particle_dist = np.linalg.norm(vel, axis=1)
            dist_to_boundary = np.linalg.norm((int_xy - pos), axis=1)

            direction = np.diagonal(np.dot((int_xy - pos), vel.T)) > 1

            # direction = (np.diag((int_xy - pos).T @ vel) /
            #             np.sum((int_xy - pos)**2, axis=0) > 1)

            reach = dist_to_boundary <= particle_dist

            dir_reach = np.logical_and(direction, reach)

            # overlap[dir_reach] = self.check_boundary_overlap(bl[dir_reach],
            #                                                  bu[dir_reach],
            #                                                  int_xy[dir_reach])
            overlap[dir_reach] = self.check_boundary_overlap(bl[dir_reach],
                                                             bu[dir_reach],
                                                             int_xy[dir_reach])

            self.x[overlap] = int_xy[:, 0][overlap]
            self.y[overlap] = int_xy[:, 1][overlap]
            self.vx[overlap] = new_vel[:, 0][overlap]
            self.vy[overlap] = new_vel[:, 1][overlap]

            if b.sticky:
                to_stick = np.random.uniform(0, 1, self.size) < b.on
                stuck = np.logical_and(overlap, to_stick)
                to_unstick = np.random.uniform(0, 1, self.size) < b.off
                unstuck = np.logical_and(~self.active, to_unstick)
                # self.x[unstuck] = (self.x[unstuck] +
                #                    (self.active[unstuck] * self.vx[unstuck]))
                # self.y[unstuck] = (self.y[unstuck] +
                #                    (self.active[unstuck] * self.vy[unstuck]))
                self.active[unstuck] = True
                self.active[stuck] = False

    def find_intersection_point(self, bl, bu):
        '''
        Calculate the point at which the particle would intersect
        with the boundary if the boundary was infinite in length
        '''
        # Check for horizontal or vertical trajectory or boundary
        p_vert = self.vx == 0
        p_hor = self.vy == 0
        p_angle = ~(np.logical_or(p_vert, p_hor))
        b_vert, b_hor = (bl[0] - bu[0]) == 0
        b_angle = not(b_vert or b_hor)

        # print(f'Particle: {p_vert}, {p_hor}, {p_angle}')
        # print(f'Boundary: {b_vert}, {b_hor}, {b_angle}')

        # Return intersection if particle trajectory is vertical or horizontal
        mp, cp = self.get_line_equ(self.x + self.vx, self.y + self.vy,
                                   self.x, self.y)
        mp[mp == 0] = 0.00001  # Used to avoid divide by zero error

        bx = bl[0, 0]
        by = bl[0, 1]
        if b_vert:
            intersection_x = bl[:, 0]
            intersection_y = ((p_vert * 100000) +
                              (p_hor * self.y) +
                              (p_angle * ((mp * bx) + cp)))
        if b_hor:
            intersection_x = ((p_vert * self.x) +
                              (p_hor * 100000) +
                              (p_angle * ((by - cp) / mp)))
            intersection_y = bl[:, 1]
        if b_angle:
            m, c = self.get_line_equ(bl[:, 0], bl[:, 1], bu[:, 0], bu[:, 1])
            m[m == 0] = 0.00001  # Used to avoid divide by zero error
            intersection_x = ((p_vert * self.x) +
                              (p_hor * (self.y - c) / m) +
                              (p_angle * ((c - cp) / (mp - m))))
            intersection_y = ((p_vert * (m * self.x) + c) +
                              (p_hor * self.y) +
                              (p_angle * (((cp * m) - (c * mp)) / (mp - m))))

        return hf.combineVectors(intersection_x, intersection_y)

    def get_line_equ(self, a_x, a_y, b_x, b_y):
        '''
        Find equations of particle tradjectory and boundary
        i.e y = mx + c
        '''
        dy = b_y - a_y
        dx = b_x - a_x
        dx[dx == 0] = 0.00001  # avoid divide-by-zero
        m = dy / dx
        c = b_y - (m * b_x)

        return m, c

    def check_boundary_overlap(self, bl, bu, int_xy):
        overlap_x = (((int_xy[:, 0] <= bu[:, 0]) *
                      (int_xy[:, 0] >= bl[:, 0])) +
                     ((bu[:, 0] <= int_xy[:, 0]) *
                      (bl[:, 0] >= int_xy[:, 0])))
        overlap_y = (((int_xy[:, 1] <= bu[:, 1]) *
                      (int_xy[:, 1] >= bl[:, 1])) +
                     ((bu[:, 1] <= int_xy[:, 1]) *
                      (bl[:, 1] >= int_xy[:, 1])))
        return (overlap_x * overlap_y)

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
        return np.count_nonzero(~self.active)

    def print_data(self):
        print('\tPos\tVel')
        print(f'\t({self.x}, {self.y})\t({self.vx}, {self.vy})')

    def plot(self, ax, history=False):
        ax.plot(self.x[self.active],
                self.y[self.active], '*')
        ax.plot(self.x[~self.active],
                self.y[~self.active],
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

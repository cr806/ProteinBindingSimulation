import numpy as np


class Particle:
    def __init__(self, initial_pos, boundaries):
        '''
        Defines attributes of the class: position, velocity,
        all previous positions and velocities
        '''
        self.active = True
        self.boundaries = boundaries
        self.pos = initial_pos
        self.velocity = np.array([0, 0])
        self.position_array = []

    def get_random_velocity(self, factor=1):
        '''
        Assign a random velocity to a particle
        '''
        velocity = factor * self.normalise_vector(np.random.uniform(-1, 1, 2))
        return np.array(velocity)

    def normalise_vector(self, vel):
        '''
        Converts a velocity vector to a unit vector
        '''
        magnitude = np.linalg.norm(vel)
        if magnitude == 0:
            return vel
        return vel / magnitude

    def update_brownian_velocity(self, factor):
        if self.active:
            self.velocity = (self.velocity +
                             self.get_random_velocity(factor=factor))

    def update_flow_velocity(self, vel):
        if self.active:
            self.velocity = self.velocity + vel

    def update_position(self):
        if self.active:
            self.pos = self.pos + self.velocity
            self.position_array.append(self.pos)
        else:
            self.position_array.append(self.pos)
        self.velocity = np.array([0, 0])

    def check_boundary(self):
        '''
        Check whether particle has interacted with a boundary
        boundary = {lower : <tuple>,
                    upper : <tuple>,
                    sticky : <bool>,
                    on : <float>,
                    off : <float>}
        '''
        for b in self.boundaries:
            if not np.all(self.velocity == 0):
                intersection = self.find_intersection_point(b)
                if not intersection:
                    # print("Particle travelling parallel to boundary")
                    continue
                intersection = np.array(intersection, dtype='object')

                # Calculate distance moved by particle and distance to
                # boundary intersection point
                particle_movement_dist = np.linalg.norm(self.pos -
                                                        (self.pos +
                                                         self.velocity))
                dist_to_boundary = np.linalg.norm(self.pos - intersection)
                if dist_to_boundary > particle_movement_dist:
                    # print("Particle doesn't reach boundary")
                    continue
                overlap = self.check_boundary_overlap(b, intersection)
                if overlap:
                    # print('Particle hits boundary')
                    if self.active:
                        self.pos = intersection
                        if b.sticky:
                            self.active = False
                        else:
                            self.velocity = -1 * self.velocity

    def find_intersection_point(self, b):
        '''
        Calculate the point at which the particle would intersect
        with the boundary if the boundary was infinite in length
        '''

        # Check for horizontal or vertical trajectory or boundary
        p_vert, p_hor = self.velocity == 0
        p_angle = not(p_vert or p_hor)
        b_vert, b_hor = ((np.array(b.lower) - np.array(b.upper)) == 0)
        b_angle = not(b_vert or b_hor)

        # Return intersection if particle trajectory is vertical or horizontal
        if p_vert:
            if b_vert:
                intersection = False
            if b_hor:
                intersection = (self.pos[0], b.lower[1])
            if b_angle:
                m, c = self.get_line_equ(b.lower, b.upper)
                intersection = (self.pos[0], (m * self.pos[0]) + c)
        if p_hor:
            if b_hor:
                intersection = False
            if b_vert:
                intersection = (b.lower[0], self.pos[1])
            if b_angle:
                m, c = self.get_line_equ(b.lower, b.upper)
                intersection = ((self.pos[1] - c) / m, self.pos[1])

        # Return intersection if boundary is vertical or horizontal
        if b_vert:
            if p_vert:
                intersection = False
            if p_hor:
                intersection = (b.lower[0], self.pos[1])
            if p_angle:
                m, c = self.get_line_equ(self.pos + self.velocity, self.pos)
                intersection = (b.lower[0],
                                (m * b.lower[0]) + c)
        if b_hor:
            if p_hor:
                intersection = False
            if p_vert:
                intersection = (self.pos[0], b.lower[1])
            if p_angle:
                m, c = self.get_line_equ(self.pos + self.velocity, self.pos)
                intersection = ((b.lower[1] - c) / m,
                                b.lower[1])

        # print('  \tVert\tHori\tAngle')
        # print(f'p:\t{p_vert}\t{p_hor}\t{p_angle}')
        # print(f'b:\t{b_vert}\t{b_hor}\t{b_angle}')
        # print(f'm: {m}, c: {c}')
        # print(f'Intersection : {intersection}')
        return intersection

    def get_line_equ(self, a, b):
        '''
        Find equations of particle tradjectory and boundary
        i.e y = mx + c
        '''

        m = (b[1] - a[1]) / (b[0] - a[0])
        c = b[1] - (m * b[0])

        return m, c

    def check_boundary_overlap(self, b, intersection):
        b_x1, b_y1 = b.lower
        b_x2, b_y2 = b.upper
        i_x, i_y = intersection

        if b_x2 >= i_x >= b_x1:
            if b_y2 >= i_y >= b_y1:
                return True

        return False

    def print_data(self):
        print('\tI-Pos\tVel\tF-Pos')
        print(f'\t{self.pos}\t{self.velocity}\t{self.pos+self.velocity}')

    def plot(self, ax, history=False):
        if self.active:
            ax.plot(self.pos[0], self.pos[1], 'b*')
        else:
            ax.plot(self.pos[0], self.pos[1], 'go')
        if history:
            data = np.array(self.position_array)
            ax.plot(data.T[0],
                    data.T[1],
                    color='blue',
                    linestyle='dotted',
                    marker='o',
                    markersize='2',
                    linewidth='0.5')

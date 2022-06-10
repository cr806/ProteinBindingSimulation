import numpy as np
import HelperFunc as hf


class Boundary:
    def __init__(self, b_l, b_u, s, on, off):
        margin = 0
        self.lower = (b_l[0]-margin, b_l[1]+margin)
        self.upper = (b_u[0]-margin, b_u[1]+margin)
        self.unit = self.get_unit(b_l, b_u)
        self.normal = self.get_normal(b_l, b_u)
        self.sticky = s
        self.on = on
        self.off = off

    def get_unit(self, b_l, b_u):
        '''
        Calculate unit vector to the supplied coords
        '''
        low = np.asarray(b_l) - np.asarray(b_l)  # Shift coords to origin
        up = np.asarray(b_u) - np.asarray(b_l)
        unit_x, unit_y = hf.normalise_vector(np.array([low[0], up[0]]),
                                             np.array([low[1], up[1]]))
        return np.asarray((unit_x[1], unit_y[1]))

    def get_normal(self, b_l, b_u):
        '''
        Calculate normal vector to the supplied coords
        '''
        unit = self.get_unit(b_l, b_u)
        normal = np.array([-unit[1], unit[0]])
        return normal

    def set_sticky(self, sticky):
        self.sticky = sticky

    def plot(self, ax):
        if self.sticky:
            ax.plot([self.lower[0], self.upper[0]],
                    [self.lower[1], self.upper[1]],
                    'r')
            # ax.plot(self.normal[0], self.normal[1], '*k')
        else:
            ax.plot([self.lower[0], self.upper[0]],
                    [self.lower[1], self.upper[1]],
                    'r--')
            # ax.plot(self.normal[0], self.normal[1], '*k')

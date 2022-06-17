import numpy as np
import HelperFunc as hf


class Boundary:
    def __init__(self, b_start, b_end, s, on, off):
        self.direction = (b_start[1] == b_end[1], b_start[0] == b_end[0])
        self.start = b_start
        self.end = b_end
        self.unit = self.get_unit(b_start, b_end)
        self.normal = self.get_normal(b_start, b_end)
        self.sticky = s
        self.on = on
        self.off = off

    def get_unit(self, b_start, b_end):
        '''
        Calculate unit vector to the supplied coords
        '''
        s = np.asarray(b_start) - np.asarray(b_start)  # Shift coords to origin
        e = np.asarray(b_end) - np.asarray(b_start)
        unit_x, unit_y = hf.normalise_vector(np.array([s[0], e[0]]),
                                             np.array([s[1], e[1]]))
        return np.asarray((unit_x[1], unit_y[1]))

    def get_normal(self, b_start, b_end):
        '''
        Calculate normal vector to the supplied coords
        '''
        unit = self.get_unit(b_start, b_end)
        normal = np.array([-unit[1], unit[0]])
        return normal

    def set_sticky(self, sticky):
        self.sticky = sticky

    def plot(self, ax):
        if self.sticky:
            ax.plot([self.start[0], self.end[0]],
                    [self.start[1], self.end[1]],
                    'r')
            # ax.plot(self.normal[0], self.normal[1], '*k')
        else:
            ax.plot([self.start[0], self.end[0]],
                    [self.start[1], self.end[1]],
                    'r--')
            # ax.plot(self.normal[0], self.normal[1], '*k')

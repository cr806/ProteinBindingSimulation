class Domain:
    def __init__(self, ll, ur):
        self.ll = ll  # lower left
        self.ur = ur  # upper right

    def check_extent(self, x, y):
        x[x < self.ll[0]] = self.ll[0] + 0
        x[x > self.ur[0]] = self.ur[0] - 0
        y[y < self.ll[1]] = self.ll[1] + 0
        y[y > self.ur[1]] = self.ur[1] - 0
        return x, y

    def plot(self, ax):
        ax.plot([self.ll[0], self.ur[0], self.ur[0], self.ur[0],
                 self.ur[0], self.ll[0], self.ll[0], self.ll[0]],
                [self.ll[1], self.ll[1], self.ll[1], self.ur[1],
                 self.ur[1], self.ur[1], self.ur[1], self.ll[1]],
                color='lightgrey',
                linewidth='2')

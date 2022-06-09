import numpy as np


def normalise_vector(vx, vy):
    '''
    Converts a velocity vector to a unit vector
    '''
    vel = np.concatenate((vx, vy)).reshape((-1, 2), order='F')
    magnitude = np.linalg.norm(vel, axis=1)
    magnitude[magnitude == 0] = 1
    return vel[:, 0] / magnitude, vel[:, 1] / magnitude


def combineVectors(v1, v2):
    v = np.concatenate((v1, v2))
    v = v.reshape((-1, 2), order='F')
    return v


def reflectVector(b_n, v):
    d = np.dot(v, b_n.T)[:, 0]
    return v - (2 * d[:, np.newaxis] * b_n)

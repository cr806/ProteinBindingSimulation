import numpy as np


def normalise_vector(vx, vy):
    '''
    Converts a velocity vector to a unit vector
    '''
    vel = np.concatenate((vx, vy)).reshape((-1, 2), order='F')
    magnitude = np.linalg.norm(vel, axis=1)
    magnitude[magnitude == 0] = 1
    return vel[:, 0] / magnitude, vel[:, 1] / magnitude


# def angle_between(v1, v2):
#     '''
#     Returns the angle in radians between vectors 'v1' and 'v2'::
#         >>> angle_between((1, 0), (0, 1))
#         1.5707963267948966
#         >>> angle_between((1, 0), (1, 0))
#         0.0
#         >>> angle_between((1, 0), (-1, 0))
#         3.141592653589793
#     '''
#     v1x, v1y = normalise_vector(v1[:, 0], v1[:, 1])
#     v1 = combineVectors(v1x, v1y)
#     v2x, v2y = normalise_vector(v2[:, 0], v2[:, 1])
#     v2 = combineVectors(v2x, v2y)
#     c = np.clip(np.dot(v1, v2.T), -1.0, 1.0)
#     c = np.diagonal(c)
#     print(f'c: {c}')
#     return np.arccos(c.T)


def combineVectors(v1, v2):
    v = np.concatenate((v1, v2))
    v = v.reshape((-1, 2), order='F')
    return v


# def rotateVector(angle, v):
#     rot = np.array([[np.cos(angle), -np.sin(angle)],
#                     [np.sin(angle), np.cos(angle)]])

#     return np.dot(v, rot.T)


# def rotateVector(angle, v):
#     rot = np.array([[np.cos(angle), -np.sin(angle)],
#                     [np.sin(angle), np.cos(angle)]])
#     # print(f'Rot: \n{rot}')
#     # print(type(rot))
#     # print(rot.shape)

    # rotated_V = np.array([rot_.dot(v_.T).T for rot_, v_ in zip(rot.T, v)])

    # return rotated_V
    # # return np.dot(v, rot.T)


def reflectVector(b_n, v):
    # vx, vy = normalise_vector(v[:, 0], v[:, 1])
    # v_ = combineVectors(vx, vy)
    # d = np.dot(v_, b_n.T)[:, 0]
    # return v_ - (2 * d[:, np.newaxis] * b_n)
    d = np.dot(v, b_n.T)[:, 0]
    return v - (2 * d[:, np.newaxis] * b_n)

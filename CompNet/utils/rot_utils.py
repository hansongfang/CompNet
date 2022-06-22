import numpy as np
import math
import re


def eulerXYZ_to_Rotmat(theta, order='XYZ'):
    rx, ry, rz = theta

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rx), -math.sin(rx)],
                    [0, math.sin(rx), math.cos(rx)]
                    ])

    R_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                    [0, 1, 0],
                    [-math.sin(ry), 0, math.cos(ry)]
                    ])

    R_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                    [math.sin(rz), math.cos(rz), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def get_rotmat_from_filename(filename):
    objNameRegex = re.compile(r'(\d+)_rx(\d+)_ry(\d+)_rz(\d+)')
    mo = objNameRegex.search(filename)
    rxyz = [int(mo.group(i)) for i in range(2, 5)]
    rxyz = np.radians(rxyz)
    rotmat = eulerXYZ_to_Rotmat(rxyz)
    return rotmat
import numpy as np
import os
import sys
import math


def quaternion2rotationmat(quatern):
    """
    translate the quaternion into rotation matrix.
    :param quatern: quaternion [w, x, y, z]
    :return: rotation matrix: 3*3
    """
    w, x, y, z = quatern
    rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y],
                             [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x],
                             [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])

    return rotation_mat


def coords_translation(y, q, t):
    """
    translate a point in left/right lidar coordinate into imu coordinate.
    Y(left/right) = RX(imu)+T,  X = R^-1 (Y-T)
    X(imu): the point in imu coords: 3*1
    Y(left/right): the point in left/right lidar coors: 3*1
    R: rotation matrix 3*3
    T: translation matrix 3*1
    :param x: points array in left/right lidar coordinate
    :param q: the quaternion to translate points from imu coords to left/right coords
    :param t: the translation parameters to translate points from imu coors to left/right coords
    :return: points in left/right coords
    """
    R = quaternion2rotationmat(q)
    X = np.dot(np.linalg.inv(R), y+t)
    # X = np.dot(R, y) + t
    # X = np.dot(np.linalg.inv(R), y+t)
    return X

def translate_points(file_path):
    """
    translate one frame points into imu coords
    :return:
    """
    # q = np.array([0.93, 0.00, -0.01, 0.37])
    # t = np.array([0.51, 1.47, 0.48])
    # q = np.array([0.3541431336629506, -0.001006951725131456, -0.009479955492482614, 0.9351426368175072])
    # q = np.array([0.917060074385124, 0.0, 0.0, 0.3987490689252462])
    # t = np.array([0.006323878187686205, 0.1167294234037399, 1.507251024246216])
    q = np.array([0.394596419887161, -0.0014195031959473004, -0.009427010026066098, 0.9188050837391357])
    t = np.array([-0.5252518057823181, 1.478088617324829, 0.5421611070632935])
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    new_points = []
    for p in points:
        new_p = coords_translation(p[:3], q, t)
        new_points.append(new_p)
    new_points = np.array(new_points)
    new_points.astype(np.float32).tofile("./bins/" + file_path.split("/")[-1])


def EulerAndQuaternionTransform(intput_data):
    data_len = len(intput_data)
    angle_is_not_rad = True

    if data_len == 3:
        r = 0
        p = 0
        y = 0
        if angle_is_not_rad:  # 180 ->pi
            r = math.radians(intput_data[0])
            p = math.radians(intput_data[1])
            y = math.radians(intput_data[2])
        else:
            r = intput_data[0]
            p = intput_data[1]
            y = intput_data[2]

        sinp = math.sin(p / 2)
        siny = math.sin(y / 2)
        sinr = math.sin(r / 2)

        cosp = math.cos(p / 2)
        cosy = math.cos(y / 2)
        cosr = math.cos(r / 2)

        w = cosr * cosp * cosy + sinr * sinp * siny
        x = sinr * cosp * cosy - cosr * sinp * siny
        y = cosr * sinp * cosy + sinr * cosp * siny
        z = cosr * cosp * siny - sinr * sinp * cosy

        return [w, x, y, z]

        # return {w, x, y, z}

    elif data_len == 4:

        w = intput_data[0]
        x = intput_data[1]
        y = intput_data[2]
        z = intput_data[3]

        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        if angle_is_not_rad:  # pi -> 180

            r = r / math.pi * 180
            p = p / math.pi * 180
            y = y / math.pi * 180

        return [r, p, y]


if __name__ == "__main__":
    # r, p, y = EulerAndQuaternionTransform([0.9275396684816124, 0.004309909016571412, -0.0108375988584172, 0.3735426556643059])
    # r, p, y = EulerAndQuaternionTransform([0.695872231545962, -0.0007232033798969326, -0.0009649166356746195, 0.7181645858607391])
    # r, p, y = EulerAndQuaternionTransform([0.3541431336629506, -0.001006951725131456, -0.009479955492482614, 0.9351426368175072])
    # print(r)  # 43.871743664412605       -0.13707755152955736
    # print(p)  # -0.005810671351988371    -0.017427042934972086
    # print(y)  # -1.3365152041541308      91.80641168688327
    # w, x, y, z = EulerAndQuaternionTransform([r, p, y-5])
    # print(w, x, y, z)
    file_path = "/home/shl/Desktop/raw_bins/"
    for f in os.listdir(file_path):
        f_name = file_path + f
        translat_points(f_name)

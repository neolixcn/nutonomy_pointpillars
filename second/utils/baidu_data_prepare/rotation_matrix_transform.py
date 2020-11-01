from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import numpy as np


def get_transform_matrix(translate,quat,inverse = True):
    rotation = R.from_quat(quat)
    matrix = np.vstack([np.hstack([rotation.as_matrix(),np.array(translate).reshape(3,1)]),np.array([0,0,0,1]).reshape(1,4)])
    if inverse:
        matrix =np.linalg.inv(matrix)
    return matrix
def merge_matrix_get_parameter(matrix_1,matrix_2):
    merge_matrix = np.dot(matrix_1, matrix_2)
    R_quat = list(Quaternion(matrix=merge_matrix[:3, :3]))
    w_r,x_r,y_r,z_r = R_quat
    x_t,y_t,z_t = merge_matrix[:3,3]
    rotation_param = [x_r,y_r,z_r,w_r]
    transform_param = [x_t,y_t,z_t]
    return transform_param,rotation_param



if __name__ == '__main__':
    camera_2_lidar = [0.6032394415412839, -0.5793481145390601, -0.1555988820876752, 0.1668704274240954 ,-0.6778271918473403, 0.6910960653988083, 0.1873253502170349]
    camera_2_lidar_matrix = get_transform_matrix(camera_2_lidar[:3],camera_2_lidar[3:])
    lidar_novatal = [0.03349882736802101 ,0.3754203915596008, 1.46821916103363, -0.0001944080312460729, -0.0006824695292739913, 0.4265022721376034, 0.9044861976393899]
    lidar_2_imu_matrix = get_transform_matrix(lidar_novatal[:3],lidar_novatal[3:],False)
    # carmera_2_imu = np.dot(lidar_2_imu_matrix,camera_2_lidar_matrix)
    # R_quat= list(Quaternion(matrix=carmera_2_imu[:3,:3]))
    # w_r,x_r,y_r,z_r = R_quat
    # x_t,y_t,z_t = carmera_2_imu[:3,3]
    transform_param,rotation_param = merge_matrix_get_parameter(lidar_2_imu_matrix, camera_2_lidar_matrix):
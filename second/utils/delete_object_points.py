import numpy as np
import os
import numba


@numba.jit(nopython=False)
def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces


@numba.jit(nopython=False)
def surface_equ_3d_jit(polygon_surfaces):
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d



@numba.jit(nopython=False)
def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jit(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                     + points[i, 1] * normal_vec[j, k, 1] \
                     + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_rbbox(points, rbbox, lidar=True):
    if lidar:
        h_axis = 2
        origin = [0.5, 0.5, 0]
    else:
        origin = [0.5, 1.0, 0.5]
        h_axis = 1
    if rbbox.shape[0] == 0:
        indices = np.ones([points.shape[0]])
    else:
        rbbox_corners = center_to_corner_box3d(
            rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=h_axis)
        surfaces = corner_to_surfaces_3d(rbbox_corners)
        indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(
        dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim])
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=[0.5, 1.0, 0.5],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def filer_boxes(label_f):
    """
    filer boxes to be deleted in one frame
    filer the box whose class is "pedestrian"  and whose distance between beginning is smaller than 5m
    :param label_f:
    :return: 3D boxes
    """
    with open(label_f, 'r') as f:
        line_ls = f.readlines()
        boxes3d = []
        for line in line_ls:
            label_ls = line.split(" ")
            object_cls = label_ls[0]
            distance_x = float(label_ls[11])
            distance_y = float(label_ls[12])
            if (int(object_cls) == 0) & (abs(distance_x) < 5) & (abs(distance_y) < 55):
                    # boxes3d.append([float(label_ls[11]), float(label_ls[12]), float(label_ls[13])+float(label_ls[8])/2,
                    #                float(label_ls[10]), float(label_ls[8]), float(label_ls[9]), -np.pi/2-float(label_ls[14])])
                    # boxes3d.append([float(label_ls[11]), float(label_ls[12]), float(label_ls[13]),
                    #                float(label_ls[10]), float(label_ls[8]), float(label_ls[9]), -np.pi/2-float(label_ls[14])])
                    boxes3d.append([float(label_ls[11]), float(label_ls[12]), float(label_ls[13]),
                                   float(label_ls[10]), float(label_ls[9]), float(label_ls[8])+1, -(-np.pi/2-float(label_ls[14]))])
        boxes3d_arr = np.array(boxes3d)
        return boxes3d_arr


def delete_one_object(boxes_3d, pc_f, new_pc_f):
    """
    delete the points of one certain object from one frame point
    :param label_f:
    :param pc_f:
    :return:
    """
    pc = np.fromfile(pc_f, dtype=np.float32).reshape(-1, 4)
    pc_mask = np.zeros([pc.shape[0]])
    point_indices = points_in_rbbox(pc, boxes_3d)
    print(len(point_indices.shape))
    if len(point_indices.shape) == 1:
        print("same same same")
        pc.astype(np.float32).tofile(new_pc_f)
    else:
        for i in range(point_indices.shape[0]):
            for j in range(point_indices.shape[-1]):
                pc_mask[i] += int(point_indices[i][j])
        pc_mask = pc_mask.astype(np.bool_)
        new_pc_mask = (pc_mask == False)
        pc_deleted = pc[new_pc_mask]
        pc_deleted.astype(np.float32).tofile(new_pc_f)


def delete_points(label_path, pc_path, new_pc_path):
    label_ls = os.listdir(label_path)
    label_ls.sort()
    for label_f in label_ls:
        print(label_f)
        boxes3d = filer_boxes(label_path + label_f)
        points_f = pc_path + label_f.strip("txt") + "bin"
        new_pc_f = new_pc_path + label_f.strip("txt") + "bin"
        # print(label_f, points_f)
        delete_one_object(boxes3d, points_f, new_pc_f)


if __name__ == "__main__":
    label_path = "/home/shl/visu_vedio/data_visual/pre_test/"
    points_path = "/home/shl/visu_vedio/data_visual/bins/"
    new_points_path = "/home/shl/visu_vedio/data_visual/deleted_bins/"
    delete_points(label_path, points_path, new_points_path)
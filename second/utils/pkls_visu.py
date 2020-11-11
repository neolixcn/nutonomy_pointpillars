# coding=utf8
import os
import numpy as np
import pickle
# from lib.net.point_rcnn import PointRCNN
# from lib.datasets.mada_rcnn_dataset import MadaRCNNDataset
# import tools.train_utils.train_utils as train_utils
# from lib.utils.bbox_transform import decode_bbox_target
# from tools.kitti_object_eval_python.visualize_common import VisualizePcd, quaternion_from_euler

# from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
# import lib.utils.kitti_utils as kitti_utils
# import lib.utils.iou3d.iou3d_utils as iou3d_utils
import logging
import math
import re
import glob
import time
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from jsk_rviz_plugins.msg import Pictogram, PictogramArray
import sys
from pynput.keyboard import Controller, Key, Listener
from pynput import keyboard
import json

# import struct

FIXED_FRAME = 'pandar'

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]


# code from /opt/ros/kinetic/lib/python2.7/dist-packages/tf/transformations.py
def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    # print("ak : {}".format(type(ak)))
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    quaternion = np.empty((4,), dtype=np.float64)
    if repetition:
        quaternion[i] = cj * (cs + sc)
        quaternion[j] = sj * (cc + ss)
        quaternion[k] = sj * (cs - sc)
        quaternion[3] = cj * (cc - ss)
    else:
        quaternion[i] = cj * sc - sj * cs
        quaternion[j] = cj * ss + sj * cc
        quaternion[k] = cj * cs - sj * sc
        quaternion[3] = cj * cc + sj * ss
    if parity:
        quaternion[j] *= -1
    return quaternion


#  /velodyne_points topic's subscriber callback function


#  publishing function for DEBUG
def publish_test(np_p_ranged, frame_id):
    header = Header()
    header.stamp = rospy.Time()
    header.frame_id = frame_id
    x = np_p_ranged[:, 0].reshape(-1)
    y = np_p_ranged[:, 1].reshape(-1)
    z = np_p_ranged[:, 2].reshape(-1)
    # if intensity field exists
    if np_p_ranged.shape[1] == 4:
        i = np_p_ranged[:, 3].reshape(-1)
    else:
        i = np.zeros((np_p_ranged.shape[0], 1)).reshape(-1)
    cloud = np.stack((x, y, z, i))
    # point cloud segments
    # 4 PointFields as channel description
    msg_segment = pc2.create_cloud(header=header,
                                   fields=_make_point_field(4),
                                   points=cloud.T)
    #  publish to /velodyne_points_modified
    point_pub.publish(msg_segment)  # DEBUG


#  code from SqueezeSeg (inspired from Durant35)
def hv_in_range(x, y, z, fov, fov_type='h'):
    """
    Extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit

    Args:
    `x`:velodyne points x array
    `y`:velodyne points y array
    `z`:velodyne points z array
    `fov`:a two element list, e.g.[-45,45]
    `fov_type`:the fov type, could be `h` or 'v',defualt in `h`

    Return:
    `cond`:condition of points within fov or not

    Raise:
    `NameError`:"fov type must be set between 'h' and 'v' "
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if fov_type == 'h':
        return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180), np.arctan2(y, x) < (-fov[0] * np.pi / 180))
    elif fov_type == 'v':
        return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), np.arctan2(z, d) > (fov[0] * np.pi / 180))
    else:
        raise NameError("fov type must be set between 'h' and 'v' ")


def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]


def gui(point, v_info, r_info, r_info2):
    new_point_ls = np.fromfile(point, dtype=np.float32).reshape([-1, 4])

    header = Header()
    header.frame_id = FIXED_FRAME
    msg_val_bboxes = BoundingBoxArray()
    msg_val_bboxes.boxes = []
    msg_val_bboxes.header = header
    msg_res_bboxes = BoundingBoxArray()
    msg_res_bboxes.boxes = []
    msg_res_bboxes.header = header
    msg_res_bboxes2 = BoundingBoxArray()
    msg_res_bboxes2.boxes = []
    msg_res_bboxes2.header = header

    print(v_info)
    val_box_num = v_info['annos']['name'].shape[0]
    dimensions = v_info['annos']['dimensions'].tolist()  # l,h,w
    locations = v_info['annos']['location'].tolist()
    rotations = v_info['annos']['rotation_y'].tolist()
    class2label = {"Pedestrian":0, "Vehicle":1, "Cyclist":2, "Unknown": 3}

    for i in range(val_box_num):
        msg_val_box = BoundingBox()
        msg_val_box.header = header
        msg_val_box.pose.position.x = locations[i][0]
        msg_val_box.pose.position.y = locations[i][1]
        msg_val_box.pose.position.z = locations[i][2] + dimensions[i][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - rotations[i])
        msg_val_box.pose.orientation.x = q[0]
        msg_val_box.pose.orientation.y = q[1]
        msg_val_box.pose.orientation.z = q[2]
        msg_val_box.pose.orientation.w = q[3]
        msg_val_box.dimensions.x = dimensions[i][0]
        msg_val_box.dimensions.y = dimensions[i][2]
        msg_val_box.dimensions.z = dimensions[i][1]
        # msg_val_box.label = class2label[v_info['name'][i]]
        msg_val_bboxes.boxes.append(msg_val_box)

    pre_box_num = r_info['name'].shape[0]
    pre_dimensions = r_info['dimensions'].tolist()
    pre_locations = r_info['location'].tolist()
    pre_rotations = r_info['rotation_y'].tolist()
    for j in range(pre_box_num):
        msg_res_box = BoundingBox()
        msg_res_box.header = header
        msg_res_box.pose.position.x = pre_locations[j][0]
        msg_res_box.pose.position.y = pre_locations[j][1]
        msg_res_box.pose.position.z = pre_locations[j][2] + pre_dimensions[j][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - pre_rotations[j])
        msg_res_box.pose.orientation.x = q[0]
        msg_res_box.pose.orientation.y = q[1]
        msg_res_box.pose.orientation.z = q[2]
        msg_res_box.pose.orientation.w = q[3]
        msg_res_box.dimensions.x = pre_dimensions[j][0]
        msg_res_box.dimensions.y = pre_dimensions[j][2]
        msg_res_box.dimensions.z = pre_dimensions[j][1]
        # msg_res_box.label = class2label[r_info['name'][j]]
        msg_res_bboxes.boxes.append(msg_res_box)



    pre_box_num2 = r_info2['name'].shape[0]
    pre_dimensions2 = r_info2['dimensions'].tolist()
    pre_locations2 = r_info2['location'].tolist()
    pre_rotations2 = r_info2['rotation_y'].tolist()
    for m in range(pre_box_num2):
        msg_res_box2 = BoundingBox()
        msg_res_box2.header = header
        msg_res_box2.pose.position.x = pre_locations2[m][0]
        msg_res_box2.pose.position.y = pre_locations2[m][1]
        msg_res_box2.pose.position.z = pre_locations2[m][2] + pre_dimensions2[m][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - pre_rotations2[m])
        msg_res_box2.pose.orientation.x = q[0]
        msg_res_box2.pose.orientation.y = q[1]
        msg_res_box2.pose.orientation.z = q[2]
        msg_res_box2.pose.orientation.w = q[3]
        msg_res_box2.dimensions.x = pre_dimensions2[m][0]
        msg_res_box2.dimensions.y = pre_dimensions2[m][2]
        msg_res_box2.dimensions.z = pre_dimensions2[m][1]
        # msg_res_box.label = class2label[r_info['name'][j]]
        msg_res_bboxes2.boxes.append(msg_res_box2)


    publish_test(new_point_ls, 'pandar')
    val_pub.publish(msg_val_bboxes)
    res_pub.publish(msg_res_bboxes)
    res_pub2.publish(msg_res_bboxes2)

    # track_pub.publish(track_picarr)
    # car_pub.publish(car_bboxes)
    # ped_pub.publish(ped_bboxes)
    # cyc_pub.publish(cyc_bboxes)
    # unk_pub.publish(unk_bboxes)


def count_label(l_path):
    classes = set()
    for l_json in os.listdir(l_path):
        with open(label_path + l_json, 'r') as f:
            labels = json.load(f)
        for label_dict in labels:
            classes.add(label_dict['type'])
    print("classes", classes)


data_id = -1


def on_press(key):
    """
    可视化验证集pkl的boxes、检测结果pkl的boxes、误检每一类的boxes漏检每一类的boxes
    :param key:
    :return:
    """
    global data_id
    # key = input()
    # key = input()
    #
    if key == Key.f1:
        data_id += 1
        # pre_f = pre_rs + pre_ls[data_id]
        # label_f = label_path + pre_ls[data_id]
        # print("label", label_f)
        # bin_f = bin_path + pre_ls[data_id].strip('txt') + 'bin'
        val_f = val_infos[data_id]
        res_f = result_infos[data_id]
        res_f2 = result_infos2[data_id]
        print(val_f['pc_idx'])
        bin_f = val_bin_path + "%06d.bin" % int(val_f['pc_idx'])
        #bin_f = val_bin_path + "%06d.bin" % int(val_f['frame_id'])
        # fn_c0 = fn_c0_dict[data_id]
        # fn_c1 = fn_c1_dict[data_id]
        # fn_c2 = fn_c2_dict[data_id]
        # fn_c3 = fn_c3_dict[data_id]
        #
        # fp_c0 = fp_c0_dict[data_id]
        # fp_c1 = fp_c1_dict[data_id]
        # fp_c2 = fp_c2_dict[data_id]
        # fp_c3 = fp_c3_dict[data_id]
        # print("fn_c0", fn_c0)
        # print("fn_c1", fn_c1)
        # print("fn_c2", fn_c2)
        # print("fn_c3", fn_c3)
        # print("fp_c0", fp_c0)
        # print("fp_c1", fp_c1)
        # print("fp_c2", fp_c2)
        # print("fp_c3", fp_c3)

        gui(bin_f, val_f, res_f, res_f2)

    elif key == 's':
        count_label(label_path)
    elif key == 'q':
        sys.exit()


def generate_fn_fp_dict(fn_file):
    """
    解析fn文件或者fp文件，得到每帧对应的fn box的index集合
    :param fn_file:fn_file: 保存fn和fp的文件，其中每一行的保存格式是 pcd_idx:3, box_id:1
    pcd_idx:fn所在的index
    box_idx:每帧中fp所在的box的index
    :return: fp（fn）的dict，key为pcd的index， value为该帧中box的index组成的集合
    """
    with open(fn_file, 'r') as f1:
        results_lines = f1.readlines()
        false_dict = {}
        for i in range(386):
            false_dict[i] = []
        for result_line in results_lines:
            false_frame_id = result_line.split(",")[0].split(":")[-1]
            false_box_id = result_line.split(",")[1].split(":")[-1].strip()
            false_dict[int(false_frame_id)].append(int(false_box_id))
        return false_dict


# 开始监听
def start_listen():
    with Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    rospy.init_node("visu", anonymous=True)
    point_pub = rospy.Publisher("pandar_points_rs", PointCloud2, queue_size=10)
    val_pub = rospy.Publisher("val_bbox", BoundingBoxArray, queue_size=10)
    res_pub = rospy.Publisher("pre_bbox", BoundingBoxArray, queue_size=10)
    res_pub2 = rospy.Publisher("pre_bbox2", BoundingBoxArray, queue_size=10)
    track_pub = rospy.Publisher("track_id", PictogramArray, queue_size=10)

    fn_ped_pub = rospy.Publisher("fn_ped_bbox", BoundingBoxArray, queue_size=10)
    fn_car_pub = rospy.Publisher("fn_car_bbox", BoundingBoxArray, queue_size=10)
    fn_cyc_pub = rospy.Publisher("fn_cyc_bbox", BoundingBoxArray, queue_size=10)
    fn_unk_pub = rospy.Publisher("fn_unk_bbox", BoundingBoxArray, queue_size=10)

    fp_ped_pub = rospy.Publisher("fp_ped_bbox", BoundingBoxArray, queue_size=10)
    fp_car_pub = rospy.Publisher("fp_car_bbox", BoundingBoxArray, queue_size=10)
    fp_cyc_pub = rospy.Publisher("fp_cyc_bbox", BoundingBoxArray, queue_size=10)
    fp_unk_pub = rospy.Publisher("fp_unk_bbox", BoundingBoxArray, queue_size=10)

    val_path = ""  ## val.pkl path
    result_path = ""  ## result.pkl path
    result_path2 = ""  ## result2.pkl path
    val_bin_path = ""  ## val points path
    f1 = open(val_path, 'rb')
    val_infos = pickle.load(f1)
    f2 = open(result_path, 'rb')
    result_infos = pickle.load(f2)
    f3 = open(result_path2, 'rb')
    result_infos2 = pickle.load(f3)
    start_listen()
    rospy.spin()





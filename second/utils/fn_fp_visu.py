#coding=utf8
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
from jsk_rviz_plugins.msg import Pictogram,PictogramArray
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
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

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
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
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
    point_pub.publish(msg_segment) #  DEBUG


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
		return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi/180), np.arctan2(y, x) < (-fov[0] * np.pi/180))
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


def gui(point, v_info, r_info, n0, n1, n2, n3, p0, p1, p2, p3):
    new_point_ls = np.fromfile(point, dtype=np.float32).reshape([-1, 4])

    header = Header()
    header.frame_id = FIXED_FRAME
    msg_val_bboxes = BoundingBoxArray()
    msg_val_bboxes.boxes = []
    msg_val_bboxes.header = header
    msg_res_bboxes = BoundingBoxArray()
    msg_res_bboxes.boxes = []
    msg_res_bboxes.header = header

    fn_ped_bboxes = BoundingBoxArray()
    fn_ped_bboxes.boxes = []
    fn_ped_bboxes.header = header
    fn_car_bboxes = BoundingBoxArray()
    fn_car_bboxes.boxes = []
    fn_car_bboxes.header = header
    fn_cyc_bboxes = BoundingBoxArray()
    fn_cyc_bboxes.boxes = []
    fn_cyc_bboxes.header = header
    fn_unk_bboxes = BoundingBoxArray()
    fn_unk_bboxes.boxes = []
    fn_unk_bboxes.header = header

    fp_ped_bboxes = BoundingBoxArray()
    fp_ped_bboxes.boxes = []
    fp_ped_bboxes.header = header
    fp_car_bboxes = BoundingBoxArray()
    fp_car_bboxes.boxes = []
    fp_car_bboxes.header = header
    fp_cyc_bboxes = BoundingBoxArray()
    fp_cyc_bboxes.boxes = []
    fp_cyc_bboxes.header = header
    fp_unk_bboxes = BoundingBoxArray()
    fp_unk_bboxes.boxes = []
    fp_unk_bboxes.header = header

    track_picarr = PictogramArray()
    track_picarr.pictograms = []
    track_picarr.header = header

    val_box_num = v_info['annos']['name'].shape[0]
    dimensions = v_info['annos']['dimensions'].tolist() #l,h,w
    locations = v_info['annos']['location'].tolist()
    rotations = v_info['annos']['rotation_y'].tolist()
    for box_id in n0:
        fn_ped_box = BoundingBox()
        fn_ped_box.header = header
        fn_ped_box.pose.position.x = locations[box_id][0]
        fn_ped_box.pose.position.y = locations[box_id][1]
        fn_ped_box.pose.position.z = locations[box_id][2] + dimensions[box_id][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - rotations[box_id])
        fn_ped_box.pose.orientation.x = q[0]
        fn_ped_box.pose.orientation.y = q[1]
        fn_ped_box.pose.orientation.z = q[2]
        fn_ped_box.pose.orientation.w = q[3]
        fn_ped_box.dimensions.x = dimensions[box_id][0]
        fn_ped_box.dimensions.y = dimensions[box_id][2]
        fn_ped_box.dimensions.z = dimensions[box_id][1]
        fn_ped_bboxes.boxes.append(fn_ped_box)
    for box_id in n1:
        fn_car_box = BoundingBox()
        fn_car_box.header = header
        fn_car_box.pose.position.x = locations[box_id][0]
        fn_car_box.pose.position.y = locations[box_id][1]
        fn_car_box.pose.position.z = locations[box_id][2] + dimensions[box_id][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - rotations[box_id])
        fn_car_box.pose.orientation.x = q[0]
        fn_car_box.pose.orientation.y = q[1]
        fn_car_box.pose.orientation.z = q[2]
        fn_car_box.pose.orientation.w = q[3]
        fn_car_box.dimensions.x = dimensions[box_id][0]
        fn_car_box.dimensions.y = dimensions[box_id][2]
        fn_car_box.dimensions.z = dimensions[box_id][1]
        fn_car_bboxes.boxes.append(fn_car_box)
    for box_id in n2:
        fn_cyc_box = BoundingBox()
        fn_cyc_box.header = header
        fn_cyc_box.pose.position.x = locations[box_id][0]
        fn_cyc_box.pose.position.y = locations[box_id][1]
        fn_cyc_box.pose.position.z = locations[box_id][2] + dimensions[box_id][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - rotations[box_id])
        fn_cyc_box.pose.orientation.x = q[0]
        fn_cyc_box.pose.orientation.y = q[1]
        fn_cyc_box.pose.orientation.z = q[2]
        fn_cyc_box.pose.orientation.w = q[3]
        fn_cyc_box.dimensions.x = dimensions[box_id][0]
        fn_cyc_box.dimensions.y = dimensions[box_id][2]
        fn_cyc_box.dimensions.z = dimensions[box_id][1]
        fn_cyc_bboxes.boxes.append(fn_cyc_box)
    for box_id in n3:
        fn_unk_box = BoundingBox()
        fn_unk_box.header = header
        fn_unk_box.pose.position.x = locations[box_id][0]
        fn_unk_box.pose.position.y = locations[box_id][1]
        fn_unk_box.pose.position.z = locations[box_id][2] + dimensions[box_id][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - rotations[box_id])
        fn_unk_box.pose.orientation.x = q[0]
        fn_unk_box.pose.orientation.y = q[1]
        fn_unk_box.pose.orientation.z = q[2]
        fn_unk_box.pose.orientation.w = q[3]
        fn_unk_box.dimensions.x = dimensions[box_id][0]
        fn_unk_box.dimensions.y = dimensions[box_id][2]
        fn_unk_box.dimensions.z = dimensions[box_id][1]
        fn_unk_bboxes.boxes.append(fn_unk_box)


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
        msg_val_bboxes.boxes.append(msg_val_box)


    res_box_num = r_info['name'].shape[0]
    dimensions = r_info['dimensions'].tolist() #l,h,w
    locations = r_info['location'].tolist()
    rotations = r_info['rotation_y'].tolist()

    for box_id in p0:
        fp_ped_box = BoundingBox()
        fp_ped_box.header = header
        fp_ped_box.pose.position.x = locations[box_id][0]
        fp_ped_box.pose.position.y = locations[box_id][1]
        fp_ped_box.pose.position.z = locations[box_id][2] + dimensions[box_id][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - rotations[box_id])
        fp_ped_box.pose.orientation.x = q[0]
        fp_ped_box.pose.orientation.y = q[1]
        fp_ped_box.pose.orientation.z = q[2]
        fp_ped_box.pose.orientation.w = q[3]
        fp_ped_box.dimensions.x = dimensions[box_id][0]
        fp_ped_box.dimensions.y = dimensions[box_id][2]
        fp_ped_box.dimensions.z = dimensions[box_id][1]
        fp_ped_bboxes.boxes.append(fp_ped_box)
    for box_id in p1:
        fp_car_box = BoundingBox()
        fp_car_box.header = header
        fp_car_box.pose.position.x = locations[box_id][0]
        fp_car_box.pose.position.y = locations[box_id][1]
        fp_car_box.pose.position.z = locations[box_id][2] + dimensions[box_id][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - rotations[box_id])
        fp_car_box.pose.orientation.x = q[0]
        fp_car_box.pose.orientation.y = q[1]
        fp_car_box.pose.orientation.z = q[2]
        fp_car_box.pose.orientation.w = q[3]
        fp_car_box.dimensions.x = dimensions[box_id][0]
        fp_car_box.dimensions.y = dimensions[box_id][2]
        fp_car_box.dimensions.z = dimensions[box_id][1]
        fp_car_bboxes.boxes.append(fp_car_box)
    for box_id in p2:
        fp_cyc_box = BoundingBox()
        fp_cyc_box.header = header
        fp_cyc_box.pose.position.x = locations[box_id][0]
        fp_cyc_box.pose.position.y = locations[box_id][1]
        fp_cyc_box.pose.position.z = locations[box_id][2] + dimensions[box_id][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - rotations[box_id])
        fp_cyc_box.pose.orientation.x = q[0]
        fp_cyc_box.pose.orientation.y = q[1]
        fp_cyc_box.pose.orientation.z = q[2]
        fp_cyc_box.pose.orientation.w = q[3]
        fp_cyc_box.dimensions.x = dimensions[box_id][0]
        fp_cyc_box.dimensions.y = dimensions[box_id][2]
        fp_cyc_box.dimensions.z = dimensions[box_id][1]
        fp_cyc_bboxes.boxes.append(fp_cyc_box)
    for box_id in p3:
        fp_unk_box = BoundingBox()
        fp_unk_box.header = header
        fp_unk_box.pose.position.x = locations[box_id][0]
        fp_unk_box.pose.position.y = locations[box_id][1]
        fp_unk_box.pose.position.z = locations[box_id][2] + dimensions[box_id][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - rotations[box_id])
        fp_unk_box.pose.orientation.x = q[0]
        fp_unk_box.pose.orientation.y = q[1]
        fp_unk_box.pose.orientation.z = q[2]
        fp_unk_box.pose.orientation.w = q[3]
        fp_unk_box.dimensions.x = dimensions[box_id][0]
        fp_unk_box.dimensions.y = dimensions[box_id][2]
        fp_unk_box.dimensions.z = dimensions[box_id][1]
        fp_unk_bboxes.boxes.append(fp_unk_box)

    for i in range(res_box_num):
        msg_res_box = BoundingBox()
        msg_res_box.header = header
        msg_res_box.pose.position.x = locations[i][0]
        msg_res_box.pose.position.y = locations[i][1]
        msg_res_box.pose.position.z = locations[i][2] + dimensions[i][1] / 2
        q = quaternion_from_euler(0, 0, -np.pi / 2 - rotations[i])
        msg_res_box.pose.orientation.x = q[0]
        msg_res_box.pose.orientation.y = q[1]
        msg_res_box.pose.orientation.z = q[2]
        msg_res_box.pose.orientation.w = q[3]
        msg_res_box.dimensions.x = dimensions[i][0]
        msg_res_box.dimensions.y = dimensions[i][2]
        msg_res_box.dimensions.z = dimensions[i][1]
        msg_res_bboxes.boxes.append(msg_res_box)



    publish_test(new_point_ls, 'pandar')
    val_pub.publish(msg_val_bboxes)
    res_pub.publish(msg_res_bboxes)
    fn_ped_pub.publish(fn_ped_bboxes)
    fn_car_pub.publish(fn_car_bboxes)
    fn_cyc_pub.publish(fn_cyc_bboxes)
    fn_unk_pub.publish(fn_unk_bboxes)

    fp_ped_pub.publish(fp_ped_bboxes)
    fp_car_pub.publish(fp_car_bboxes)
    fp_cyc_pub.publish(fp_cyc_bboxes)
    fp_unk_pub.publish(fp_unk_bboxes)



    # track_pub.publish(track_picarr)
    # car_pub.publish(car_bboxes)
    # ped_pub.publish(ped_bboxes)
    # cyc_pub.publish(cyc_bboxes)
    # unk_pub.publish(unk_bboxes)
    
def count_label(l_path):
    classes = set()
    for l_json in os.listdir(l_path):
        with open(label_path+l_json, 'r') as f:
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
        bin_f = val_bin_path + val_f['velodyne_path'].split("/")[-1]
        print(bin_f)

        fn_c0 = fn_c0_dict[data_id]
        fn_c1 = fn_c1_dict[data_id]
        fn_c2 = fn_c2_dict[data_id]
        fn_c3 = fn_c3_dict[data_id]

        fp_c0 = fp_c0_dict[data_id]
        fp_c1 = fp_c1_dict[data_id]
        fp_c2 = fp_c2_dict[data_id]
        fp_c3 = fp_c3_dict[data_id]
        print("fn_c0", fn_c0)
        print("fn_c1", fn_c1)
        print("fn_c2", fn_c2)
        print("fn_c3", fn_c3)
        print("fp_c0", fp_c0)
        print("fp_c1", fp_c1)
        print("fp_c2", fp_c2)
        print("fp_c3", fp_c3)

        gui(bin_f, val_f, res_f, fn_c0, fn_c1, fn_c2, fn_c3, fp_c0, fp_c1, fp_c2, fp_c3)

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
    track_pub = rospy.Publisher("track_id", PictogramArray, queue_size=10)

    fn_ped_pub = rospy.Publisher("fn_ped_bbox", BoundingBoxArray, queue_size=10)
    fn_car_pub = rospy.Publisher("fn_car_bbox", BoundingBoxArray, queue_size=10)
    fn_cyc_pub = rospy.Publisher("fn_cyc_bbox", BoundingBoxArray, queue_size=10)
    fn_unk_pub = rospy.Publisher("fn_unk_bbox", BoundingBoxArray, queue_size=10)

    fp_ped_pub = rospy.Publisher("fp_ped_bbox", BoundingBoxArray, queue_size=10)
    fp_car_pub = rospy.Publisher("fp_car_bbox", BoundingBoxArray, queue_size=10)
    fp_cyc_pub = rospy.Publisher("fp_cyc_bbox", BoundingBoxArray, queue_size=10)
    fp_unk_pub = rospy.Publisher("fp_unk_bbox", BoundingBoxArray, queue_size=10)


    val_path = "/home/shl/data/performance_for_imu_back_lidar/compare_pkls/back_lidar_infos_val.pkl"
    result_path = "/home/shl/data/performance_for_imu_back_lidar/compare_pkls/result.pkl"
    val_bin_path = "/data/data/dataset/shanghai_to_annotate/2614_back_lidar/imu_back_bin/"
    f1 = open(val_path, 'rb')
    val_infos = pickle.load(f1)
    f2 = open(result_path, 'rb')
    result_infos = pickle.load(f2)

    fn_class0_path = "/home/shl/data/performance_for_imu_back_lidar/fn.class_0.overlap1.txt"
    fn_class1_path = "/home/shl/data/performance_for_imu_back_lidar/fn.class_1.overlap1.txt"
    fn_class2_path = "/home/shl/data/performance_for_imu_back_lidar/fn.class_2.overlap1.txt"
    fn_class3_path = "/home/shl/data/performance_for_imu_back_lidar/fn.class_3.overlap1.txt"

    fp_class0_path = "/home/shl/data/performance_for_imu_back_lidar/fp.class_0.overlap1.txt"
    fp_class1_path = "/home/shl/data/performance_for_imu_back_lidar/fp.class_1.overlap1.txt"
    fp_class2_path = "/home/shl/data/performance_for_imu_back_lidar/fp.class_2.overlap1.txt"
    fp_class3_path = "/home/shl/data/performance_for_imu_back_lidar/fp.class_3.overlap1.txt"
    fn_c0_dict = generate_fn_fp_dict(fn_class0_path)
    fn_c1_dict = generate_fn_fp_dict(fn_class1_path)
    fn_c2_dict = generate_fn_fp_dict(fn_class2_path)
    fn_c3_dict = generate_fn_fp_dict(fn_class3_path)

    fp_c0_dict = generate_fn_fp_dict(fp_class0_path)
    fp_c1_dict = generate_fn_fp_dict(fp_class1_path)
    fp_c2_dict = generate_fn_fp_dict(fp_class2_path)
    fp_c3_dict = generate_fn_fp_dict(fp_class3_path)


    start_listen()
    rospy.spin()





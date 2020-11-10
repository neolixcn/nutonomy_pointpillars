#coding=utf8
import os
import numpy as np
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
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import sys
# from pynput.keyboard import Controller, Key, Listener
# from pynput import keyboard
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
    msg_segment = pc2.create_cloud(header=header, fields=_make_point_field(4), points=cloud.T)
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


def generate_box_from_points(points_ls):
    """
    generate 3d box from 8 points
    :param points_ls:
    p: [x, y, z]
    :return: 3d box [x, y ,z, l, w, h, theta]
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = points_ls.tolist()
    xmean = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
    ymean = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
    zmean = (p1[2] + p5[2]) / 2
    l = np.sqrt(np.power((p1[0] - p2[0]), 2) + np.power((p1[1] - p2[1]), 2))
    w = np.sqrt(np.power((p1[0] - p4[0]), 2) + np.power((p1[1] - p4[1]), 2))
    h = abs(p5[2] - p1[2])

    x_mean_12 = (p1[0] + p2[0]) / 2
    y_mean_12 = (p1[1] + p2[1]) / 2
    # theta = np.arctan2(x_mean_12, y_mean_12)
    print("arccos", (p1[0] + p2[0] - 2 * xmean) / w)
    theta = np.arccosh((p1[0] + p2[0] - 2 * xmean) / w)
    print(theta)
    box3d = np.array([xmean, ymean, zmean, l, w, h, theta])
    return box3d


def pc8_lines(pc_ls, cls, header, obj_index):
    """
    change 8 points into markerarray lines
    1--> 2, 4, 5
    2--> 3, 6
    3--> 4, 7
    4--> 8
    5--> 6, 8
    6--> 7
    7--> 8
    :param pc_ls:
    :param header:
    :return:
    """
    msg_ls = []
    for i in range(12):
        msg_line = Marker()
        msg_line.header = header
        msg_line.ns = "line"
        msg_line.action = 0#Marker.ADD #0
        msg_line.pose.orientation.z = 1.0
        msg_line.type = Marker.LINE_LIST
        msg_line.scale.x = 0.1
        msg_line.color.a = 1.0
        msg_line.color.r = 1  ## to be modify
        msg_line.id = obj_index*12 + i
        msg_line.points.clear()

        msg_ls.append(msg_line)

    pc_ls[:, 2] = pc_ls[:, 2]
    pc_ls = (-pc_ls).tolist()
    # msg_l0 msg_ls[0]
    p1 = Point(pc_ls[0][0], pc_ls[0][1], pc_ls[0][2])
    p2 = Point(pc_ls[1][0], pc_ls[1][1], pc_ls[1][2])
    msg_ls[0].points.append(p1)
    msg_ls[0].points.append(p2)

    p1 = Point(pc_ls[0][0], pc_ls[0][1], pc_ls[0][2])
    p2 = Point(pc_ls[3][0], pc_ls[3][1], pc_ls[3][2])
    msg_ls[1].points.append(p1)
    msg_ls[1].points.append(p2)

    p1 = Point(pc_ls[0][0], pc_ls[0][1], pc_ls[0][2])
    p2 = Point(pc_ls[4][0], pc_ls[4][1], pc_ls[4][2])
    msg_ls[2].points.append(p1)
    msg_ls[2].points.append(p2)

    p1 = Point(pc_ls[1][0], pc_ls[1][1], pc_ls[1][2])
    p2 = Point(pc_ls[2][0], pc_ls[2][1], pc_ls[2][2])
    msg_ls[3].points.append(p1)
    msg_ls[3].points.append(p2)

    p1 = Point(pc_ls[1][0], pc_ls[1][1], pc_ls[1][2])
    p2 = Point(pc_ls[5][0], pc_ls[5][1], pc_ls[5][2])
    msg_ls[4].points.append(p1)
    msg_ls[4].points.append(p2)

    p1 = Point(pc_ls[2][0], pc_ls[2][1], pc_ls[2][2])
    p2 = Point(pc_ls[3][0], pc_ls[3][1], pc_ls[3][2])
    msg_ls[5].points.append(p1)
    msg_ls[5].points.append(p2)

    p1 = Point(pc_ls[2][0], pc_ls[2][1], pc_ls[2][2])
    p2 = Point(pc_ls[6][0], pc_ls[6][1], pc_ls[6][2])
    msg_ls[6].points.append(p1)
    msg_ls[6].points.append(p2)

    p1 = Point(pc_ls[3][0], pc_ls[3][1], pc_ls[3][2])
    p2 = Point(pc_ls[7][0], pc_ls[7][1], pc_ls[7][2])
    msg_ls[7].points.append(p1)
    msg_ls[7].points.append(p2)

    p1 = Point(pc_ls[4][0], pc_ls[4][1], pc_ls[4][2])
    p2 = Point(pc_ls[5][0], pc_ls[5][1], pc_ls[5][2])
    msg_ls[8].points.append(p1)
    msg_ls[8].points.append(p2)

    p1 = Point(pc_ls[4][0], pc_ls[4][1], pc_ls[4][2])
    p2 = Point(pc_ls[7][0], pc_ls[7][1], pc_ls[7][2])
    msg_ls[9].points.append(p1)
    msg_ls[9].points.append(p2)

    p1 = Point(pc_ls[5][0], pc_ls[5][1], pc_ls[5][2])
    p2 = Point(pc_ls[6][0], pc_ls[6][1], pc_ls[6][2])
    msg_ls[10].points.append(p1)
    msg_ls[10].points.append(p2)

    p1 = Point(pc_ls[6][0], pc_ls[6][1], pc_ls[6][2])
    p2 = Point(pc_ls[7][0], pc_ls[7][1], pc_ls[7][2])
    msg_ls[11].points.append(p1)
    msg_ls[11].points.append(p2)

    return msg_ls







    # line_list.color.r = 1.0;
    # line_list.color.a = 1.0;
    # // Create the vertices for the points and lines
    # geometry_msgs::Point p;
    # p.x = f;
    # p.y = 0;
    # p.z = 0;
    # // The line list needs two points for each line
    # line_list.points.push_back(p);
    # p.z += 3.0;
    # line_list.points.push_back(p);





def gui_24d_label(point, label, img_f):
    new_point_ls = np.fromfile(point, dtype=np.float32).reshape([-1, 4])
    header = Header()
    header.frame_id = FIXED_FRAME
    msg_gt_markerarr = MarkerArray()
    msg_gt_markerarr.markers = []


    label_arr = np.fromfile(label, dtype=np.float32).reshape(-1, 25)
    obj_num = label_arr.shape[0]
    cls_arr = label_arr[:, 24]
    msg_gt_markerarr.markers = []
    for i in range(obj_num):
        # msg_marker = Marker()
        # msg_marker.header = header
        box_points = label_arr[i][:24].reshape(8, 3)
        # x, y, z, l, w, h, theta = generate_box_from_points(box_points)
        marker_lines = pc8_lines(box_points, cls_arr[i], header, i)
        for i in range(12):
            msg_gt_markerarr.markers.append(marker_lines[i])
    publish_test(new_point_ls, 'pandar')
    gt_markerarr_pub.publish(msg_gt_markerarr)






def gui_7d_label(point, label_f, img_f):
    new_point_ls = np.fromfile(point, dtype=np.float32).reshape([-1, 4])

    header = Header()
    header.frame_id = FIXED_FRAME
    msg_pred_bboxes = BoundingBoxArray()
    msg_pred_bboxes.boxes = []
    msg_pred_bboxes.header = header
    msg_gt_bboxes = BoundingBoxArray()
    msg_gt_bboxes.boxes = []
    msg_gt_bboxes.header = header

    img_temp = Image()
    img_temp.header = header
    img_temp.encoding = 'rgb8'
    img_data = cv2.imread(img_f)
    # img_temp.data = np.array(img_data).tostring()
    bridge = CvBridge()
    img_msg = bridge.cv2_to_imgmsg(img_data, encoding="bgr8")
    # image_temp.height = 1080
    # img_temp.width = 1920

    label_ls = np.fromfile(label_f, dtype=np.float32).reshape(-1, 8)
    for label in label_ls:
        msg_gt_box = BoundingBox()
        msg_gt_box.header = header
        cls = int(label[-1])
        x, y, z, l, w, h, theta = label[:7]
        z = 0 + h / 2
        msg_gt_box.pose.position.x = x
        msg_gt_box.pose.position.y = y
        msg_gt_box.pose.position.z = z
        q = quaternion_from_euler(0, 0, theta)
        msg_gt_box.pose.orientation.x = q[0]
        msg_gt_box.pose.orientation.y = q[1]
        msg_gt_box.pose.orientation.z = q[2]
        msg_gt_box.pose.orientation.w = q[3]
        msg_gt_box.dimensions.x = l
        msg_gt_box.dimensions.y = w
        msg_gt_box.dimensions.z = h
        msg_gt_box.label = cls
        msg_gt_bboxes.boxes.append(msg_gt_box)


    # with open(label, 'r') as f_l:
    #     labels_ls = f_l.readlines()
    # # for l_ls in labels_ls:
    # #     new_label_ls.append(l_ls.strip().split(" "))
    # # classes_value = {"Pedestrian" : 0, "Vehicle" : 1, "Cyclist" : 2, "Unknown" : 3, "Others_moving" : 4, "Non_vehicle" : 5, "Others_stationary" : 6, "Car" : 7, "DontCare" : 8, "Others" : 9, "Van": 10, "Bus": 11, "Bicycle_no_person":12, "Bicycles":13}
    # # classes_value = {"adult": 0, "animal": 1, "barrier": 2, "bicycle": 3, "bicycles": 4, "bus": 5, "car": 6, "child": 7, "cyclist": 8, "dontcare": 9,
    # #                "motorcycle": 10, "motorcyclist": 11, "tricycle": 12, "truck": 13}
    # classes_value = {'Car':0, 'Others_stationary':1, 'Non_vehicle':2, 'Vehicle':3, 'Others':4, 'Pedestrian':5, 'Others_moving':6, 'Cyclist':7}
    #
    # if labels_ls is not None:
    #     for b in labels_ls:
    #         b = b.strip().split(" ")
    #         msg_gt_box = BoundingBox()
    #         msg_gt_box.header = header
    #         #b[0] = classes_value[b[0]]
    #         b[0] = 0 + 6
    #         # b[0] = int(b[0])
    #         b[1:] = np.array(b[1:]).astype(np.float32)
    #         # b = np.array(b).astype(np.float32)
    #         # camera to lidar: x=z; y=-x; z=-y; az = pi/2 - ay
    #         msg_gt_box.pose.position.x = b[11]#b[11]
    #         msg_gt_box.pose.position.y = b[12]#b[12]
    #         msg_gt_box.pose.position.z = b[13]+b[8]/2
    #         q = quaternion_from_euler(0, 0, -np.pi/2-(b[14]))
    #         msg_gt_box.pose.orientation.x = q[0]
    #         msg_gt_box.pose.orientation.y = q[1]
    #         msg_gt_box.pose.orientation.z = q[2]
    #         msg_gt_box.pose.orientation.w = q[3]
    #
    #         msg_gt_box.dimensions.x = b[10]
    #         msg_gt_box.dimensions.y = b[9]
    #         msg_gt_box.dimensions.z = b[8]
    #         msg_gt_box.label = int(b[0])
    #         msg_gt_bboxes.boxes.append(msg_gt_box)

    publish_test(new_point_ls, 'pandar')
    gt_pub.publish(msg_gt_bboxes)
    img_pub.publish(img_msg)

    
def count_label(l_path):
    classes = set()
    for l_json in os.listdir(l_path):
        with open(label_path+l_json, 'r') as f:
            labels = json.load(f)
        for label_dict in labels:
            classes.add(label_dict['type'])


def rename_detection_folder(detection_path, new_detection_path):
    dt_ls = os.listdir(detection_path)
    dt_ls.sort()
    for dt in dt_ls:
        raw_name = detection_path + dt
        new_name = new_detection_path + dt.split("_")[2] + ".bin"
        os.rename(raw_name, new_name)


if __name__ == "__main__":
    # dt_name = "/home/shl/visu_vedio/data_visual/detection_distance0.4/detection/"
    # new_dt_name = "/home/shl/visu_vedio/data_visual/detection_distance0.4/new_detection/"
    # rename_detection_folder(dt_name, new_dt_name)
    rospy.init_node("visu", anonymous=True)
    point_pub = rospy.Publisher("pandar_points_rs", PointCloud2, queue_size=1)
    # pred_pub = rospy.Publisher("pre_bbox", BoundingBoxArray, queue_size=1)
    gt_pub = rospy.Publisher("gt_bbox", BoundingBoxArray, queue_size=1)
    gt_markerarr_pub = rospy.Publisher("gt_marker_line",  MarkerArray, queue_size=1)
    # track_pub = rospy.Publisher("track_id", PictogramArray, queue_size=1)
    img_pub = rospy.Publisher("img", Image, queue_size=1)

    label_path = "/home/shl/visu_vedio/data_visual/detection_8d/deleted_new_detection/"
    bin_path = "/home/shl/visu_vedio/data_visual/deleted_bins/"
    # pre_rs = "/home/shl/visu_vedio/data_visual/pre_test/"
    img_path = "/home/shl/visu_vedio/data_visual/new_png/"
    label_ls = os.listdir(label_path)
    # pre_ls = os.listdir(pre_rs)
    label_ls.sort()
    # print(label_ls)
    data_id = -1
    while(True):
        data_id += 1
        # pre_f = pre_rs + pre_ls[data_id]
        label_f = label_path + label_ls[data_id]
        # print("label", label_f)
        bin_f = bin_path + label_ls[data_id].strip('bin') + 'bin'
        img_f = img_path + label_ls[data_id].strip('bin') + 'png'
        # gui_24d_label(bin_f, label_f, img_f)
        gui_7d_label(bin_f, label_f, img_f)
        # time.sleep(10)

    start_listen()
    rospy.spin()





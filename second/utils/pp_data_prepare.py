import os
import numpy as np
import json
import shutil
import math
import copy
import glob
from collections import Counter
import fire

def pcd2bin(pp, bp):
    """
    transform .pcd into .bin
    """
    rs_ls = []
    with open(pp, 'r') as f:
        raw_rs = f.readlines()[11:]
    for p_line in raw_rs:
        p_ls = p_line.strip().split(" ")
        if (p_ls[0] != "nan") | (p_ls[1] != "nan") | (p_ls[2] != "nan"):
            # rs_ls.append([float(i) for i in p_ls[:3]] + [float(p_ls[3]) / 255])
            rs_ls.append([float(i) for i in p_line.strip().split(" ")[:3]] + [0.0])
            # rs_ls.append([float(i) for i in p_line.strip().split(" ")[:3]])
    np.array(rs_ls).astype(np.float32).tofile(bp)

def pcds2bins(pcd_p, bin_p):
    for pcd in os.listdir(pcd_p):
        pp = pcd_p + pcd
        bp = bin_p + pcd.strip('pcd') + 'bin'
        pcd2bin(pp, bp)


def correct_label(raw_label_path, new_label_path):
    label_ls = os.listdir(raw_label_path)
    label_ls.sort()
    for label_f in label_ls:
        label_str_ls = []
        label_name = raw_label_path + label_f
        with open(label_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_ls = line.split(" ")
                if line_ls[0] == "0":
                    line_ls[0] = "Pedestrian"
                elif line_ls[0] == "1":
                    line_ls[0] = "Vehicle"
                elif line_ls[0] == "2":
                    line_ls[0] = "Cyclist"
                elif line_ls[0] == "3":
                    line_ls[0] = "Unknown"
                label_str_ls.append(" ".join(line_ls))
        with open(new_label_path + label_f, 'w') as f1:
            f1.write("".join(label_str_ls))


def bin2pcd(bp, pp):
    """
    transform .bin ino .pcd
    """
    points = np.fromfile(bp, dtype=np.float32).reshape(-1, 4)
    pcd_ls = ["# .PCD v0.7 - Point Cloud Data file format", "VERSION 0.7", "FIELDS x y z rgb", "SIZE 4 4 4 4",
              "TYPE F F F U", "COUNT 1 1 1 1", "WIDTH 28800", "HEIGHT 1", "VIEWPOINT 0 0 0 1 0 0 0", "POINTS 28800", "DATA ascii"]
    for i in range(points.shape[0]):
        pcd_ls.append(" ".join([str(points[i][0]), str(points[i][1]), str(points[i][2]), str(points[i][3])]))
    with open(pp, 'w') as f:
        f.write("\n".join(pcd_ls))



def bins2pcds(bin_p, pcd_p):
    for bi in os.listdir(bin_p):
        bp = bin_p + bi
        pp = pcd_p + bi.strip('bin') + 'pcd'
        bin2pcd(bp, pp)


def json2txt(jp, tp):
    """
    transform .json into .txt
    json format:
    [{'cube_index': '', 'dirConfirm': True, 'globalID': 'b5c74eea41fb4d70b0c0d1d26c6e6e34', 'pointNum': 187,
     'position': {'x': '-1.14683', 'y': '-6.95712', 'z': '0.521688'}, 'rotation': {'phi': '1.49084', 'psi': '0', 'theta': '0'},
     'size': ['0.384084', '0.597939', '1.82432'], 'type': '行人'},
    txt format:
    Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.1772 0.562968 0.624843 2.93792 3.8203 0.370898 -2.98323
    """
    with open(jp, 'r') as f:
        label = json.load(f)
        label_content = ""
        with open(tp, 'w') as f:
            for l in label:
                if l['type'] == "行人":
                    cls = "Pedestrian"
                elif l['type'] == "车辆":
                    cls = "Car"
                elif l['type'] == "非机动车":
                    cls = "Non_vehicle"
                elif l['type'] == "其他":
                    cls = "Others"
                elif l['type'] == "其他（可移动）":
                    cls = "Others_moving"
                elif l['type'] == "二轮车":
                    cls = "Cyclist"
                elif l['type'] == "其他（静止）":
                    cls = "Others_stationary"
                elif l['type'] == "机动车":
                    cls = "Vehicle"
                    # "Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01"
                    # kitti label  format, 位置坐标是雷达底面坐标， 方向是相机坐标系下的方向，ry=-pi/2-rz
                label_content += cls + " " + "0.00 0 -0.20 712.40 143.00 810.73 307.92 " + l["size"][2] + " "\
                                 + l["size"][1] + " " + l["size"][0] + " " + l["position"]['x'] + " "\
                                 + l["position"]['y'] + " " + str(float(l["position"]['z']) - float(l["size"][2]) / 2) \
                                 + " " + str(-np.pi/2-float(l["rotation"]['phi'])) + '\n'
            label_content = label_content.strip()
            f.write(label_content)


def jsons2txts(json_p, txt_p):
    for js in os.listdir(json_p):
        jp = json_p + js
        tp_name = "%06d.txt" % int(js.strip('.json'))
        print(tp_name)
        tp = txt_p + tp_name
        json2txt(jp, tp)



def labels2txt2(label_p, txt_p):
    for lp in os.listdir(label_p):
        lp = label_p + lp
        tp_name = lp.strip("label") + "txt"
        print(tp_name)
        tp = txt_p + tp_name
        json2txt(lp, tp)


def count_labels(l_path, class_num):
    """
    size: h,w,l
    z_axis: the coordinate of the bounding box in z-axis
    """
    classes_4 = ['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown']
    classes_8 = ["Vehicle", "Pedestrian", "Cyclist", "Unknown", "Tricycle", "Bicycles", "Barrier", "Bicycle"]
    classes_8_chaoyang_park = ["Pedestrian", "Car", "Non_vehicle", "Others", "Others_moving", "Cyclist", "Others_stationary", "Vehicle"]
    if class_num == 4:
        classes_ls = classes_4
    elif class_num == 8:
        classes_ls = classes_8
    elif class_num == 7:
        classes_ls = classes_8_chaoyang_park
    else:
        print("error class number!")
    for cls in classes_ls:
        size = []
        z_axis = []
        for l in os.listdir(l_path):
            with open(l_path + l) as f:
                label_lines = f.readlines()
                for label_line in label_lines:
                    label_line = label_line.split(" ")
                    if label_line[0] == cls:
                        size.append([float(label_line[8]), float(label_line[9]), float(label_line[10])])
                        z_axis.append(float(label_line[13]))
                    # if (float(label_line[8])>3)|(float(label_line[9])>3)|(float(label_line[10])>3):
                    #     print("the vehicle is in %s" % l_path+l)
        np_size = np.array(size)
        np_z_axis = np.array(z_axis)
        print(np_size.shape)
        if np_size.shape[0] != 0:
            print("the number of %s is %d" % (cls, np_size.shape[0]))
            print("the mean height of %s" % cls, np_size[:, 0].mean())
            print("the mean width of %s" % cls, np_size[:, 1].mean())
            print("the mean length of %s" % cls, np_size[:, 2].mean())
            print("the mean z coordinate of %s" % cls, np_z_axis.mean())



def syn_lidar_image(raw_camera, raw_lidar):
    img_ls = os.listdir(raw_camera)
    img_ls.sort()
    lidar_ls = os.listdir(raw_lidar)
    lidar_ls.sort()
    frame_num = len(lidar_ls)
    for i in range(frame_num):
        raw_name_img = raw_camera + img_ls[i]
        new_name_img = raw_camera + "%06d.png" % i
        raw_name_lidar = raw_lidar + lidar_ls[i]
        new_name_lidar = raw_lidar + "%06d.pcd" % i
        os.rename(raw_name_img, new_name_img)
        os.rename(raw_name_lidar, new_name_lidar)


def filter_lidar(raw_path, new_path):
    img_ls = os.listdir(raw_path)
    img_ls.sort()
    for i in range(513):
        shutil.copy(raw_path+img_ls[i * 10], new_path)

def rename_file(img_path, lidar_path):
    img_ls = os.listdir(img_path)
    lidar_ls = os.listdir(lidar_path)
    img_ls.sort()
    lidar_ls.sort()
    with open("file_name_timestamps.txt", 'w') as f1:
        f1.write("\n".join(lidar_ls))

    frame_num = len(img_ls)
    for i in range(frame_num):
        print(i)
        raw_img_name = img_path + img_ls[i]
        new_img_name = img_path + "%06d.png" % (i + 3026)
        raw_lidar_name = lidar_path + lidar_ls[i]
        new_lidar_name = lidar_path + "%06d.pcd" % (i + 3026)
        os.rename(raw_img_name, new_img_name)
        os.rename(raw_lidar_name, new_lidar_name)


def rename_one_folder(file_path):
    fl_ls = os.listdir(file_path)
    fl_ls.sort()
    frame_num = len(fl_ls)
    for i in range(frame_num):
        raw_fl_name = file_path + fl_ls[i]
        new_fl_name = file_path + "%06d.bin" % (int(fl_ls[i].strip(".txt"))-100000)
        # new_fl_name = file_path + "%06d.txt" % (110000 + i)
        os.rename(raw_fl_name, new_fl_name)



def copy_file(img_path, raw_pcd_path, new_pcd_path):
    for img in os.listdir(img_path):
        raw_pcd_name = raw_pcd_path + img.strip("png") + "pcd"
        shutil.copy(raw_pcd_name, new_pcd_path)

def euler_from_quaternion(quaternion):
    x, y, z, w = quaternion
    print("quaternion", quaternion)
    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    r = r / math.pi * 180
    p = math.asin(2 * (w * y - z * x))
    p = p / math.pi * 180
    y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    y = y / math.pi * 180
    return r, p, y

def generate_graviti_label_shibiao(label_path):
    f = open(label_path)
    rs = json.load(f)
    for frame_info in rs['pictures']:
        frame_name = "./annotation_txt/" + frame_info['fileuri'].strip(".pnts").split("/")[-1] + '.txt'
        frame_info = frame_info['labelInfos']
        frame_ls = []
        for obj_info in frame_info:
            position = obj_info['position']
            quaternion = obj_info['quaternion']
            rotation = euler_from_quaternion(quaternion)[2]
            rotation = rotation / 180 * np.pi
            scale = obj_info['scale']
            cls = obj_info['target']['nameEn']
            visibility = obj_info['target']['attributes'][0]['value']
            obj_ls = [cls, '0', str(visibility), '0', '0.0 0.0 0.0 0.0', str(scale[2]), str(scale[1]), str(scale[0]),
                      str(position[0]), str(position[1]), str(position[2] - scale[2] / 2), str(-np.pi/2-rotation)]
            obj_str = " ".join(obj_ls)
            frame_ls.append(obj_str)
        frame_str = "\n".join(frame_ls)
        print(frame_str)
        with open(frame_name, 'w') as f:
            f.write(frame_str)

def generate_graviti_label(label_path):
    f = open(label_path)
    rs = json.load(f)
    for frame_info in rs['annotation']:
        frame_name = "./part5/annotation_txt/" + "%06d.txt" % (int(frame_info['fileuri'].strip(".pnts").split("/")[-1]))
        frame_info = frame_info['labels_box3D']
        frame_ls = []
        count = 0
        for obj_info_all in frame_info:
            obj_info = obj_info_all["box3D"]
            position_dict = obj_info['translation']
            position = [position_dict["x"], position_dict["y"], position_dict["z"]]
            count +=1
            print("frame_name", frame_name)
            if position[0] == None or position[1] == None or position[2] == None:
                print("error file:", frame_name)
                print("error object:", count)
                continue
            quaternion_dict = obj_info['rotation']
            quaternion = [quaternion_dict['x'], quaternion_dict['y'], quaternion_dict['z'], quaternion_dict['w']]
            rotation = euler_from_quaternion(quaternion)[2]
            rotation = rotation / 180 * np.pi
            scale_dict = obj_info['size']
            scale = [scale_dict['x'], scale_dict['y'], scale_dict['z']]
            cls = obj_info_all['category']
            print("obj_info_all['attributes']", obj_info_all['attributes'])
            visibility = obj_info_all['attributes']['visibility']
            obj_ls = [cls, '0', str(visibility), '0', '0.0 0.0 0.0 0.0', str(scale[2]), str(scale[1]), str(scale[0]),
                      str(position[0]), str(position[1]), str(position[2] - scale[2] / 2), str(-np.pi/2-rotation)]
            obj_str = " ".join(obj_ls)
            frame_ls.append(obj_str)
        frame_str = "\n".join(frame_ls)
        # print(frame_str)
        with open(frame_name, 'w') as f:
            f.write(frame_str)


def copy_bin_by_pcd(pcd_path, bin_path, new_bin_path):
    for pcd_f in os.listdir(pcd_path):
        bin_name = bin_path + pcd_f.strip('pcd') + 'bin'
        shutil.copy(bin_name, new_bin_path)


def rename_copyed_bins(bin_path):
    f_name = os.listdir(bin_path)
    f_name.sort()
    for i in range(len(f_name)):
        new_name = bin_path + "%06d.txt" % i
        os.rename(bin_path + f_name[i], new_name)

def order_file(file_path):
    rs_ls = os.listdir(file_path)
    rs_ls.sort()
    frame_num = len(rs_ls)
    for i in range(frame_num):
        raw_path = file_path + rs_ls[i]
        new_path = file_path + "%06d.txt" % i
        os.rename(raw_path, new_path)
        

def get_class_name(label_path):
    class_set = set([])
    for f in os.listdir(label_path):
        with open(label_path+f, 'r') as f:
            rs_lines = f.readlines()
            for rs_line in rs_lines:
                class_set.add(rs_line.split(" ")[0])
    print(class_set)

def copy_4pcds_from_img(img_p, left_raw, right_raw, back_raw, compen_raw,
                                          left_new, right_new, back_new, compen_new, time_stamp_txt=None, parser_with_5ms=True):
    if time_stamp_txt is None:
        img_file_name = os.listdir(img_p)
    else:
        with open(time_stamp_txt, 'r') as time_f:
            line_ls = time_f.readlines()
            back_lidar_file_name = []
            for line in line_ls:
                back_lidar_file_name.append(line.strip())
    img_file_name.sort()
    ignore_img_ls = []
    if not parser_with_5ms:
        left_lidar_file = os.listdir(left_raw)
        left_lidar_file.sort()
        right_lidar_file = os.listdir(right_raw)
        right_lidar_file.sort()
        back_lidar_file = os.listdir(back_raw)
        back_lidar_file.sort()
        compen_lidar_file = os.listdir(compen_raw)
        compen_lidar_file.sort()
        for img_file in img_file_name:
            for left_lidar in left_lidar_file:
                left_flag = time_stamp_compare(img_file, 1, 2, left_lidar, 1, 2)
                if left_flag:
                    for right_lidar in right_lidar_file:
                        right_flag = time_stamp_compare(img_file, 1, 2, right_lidar, 1, 2)
                        if right_flag:
                            for back_lidar in back_lidar_file:
                                back_flag = time_stamp_compare(img_file, 1, 2, back_lidar, 1, 2)
                                if back_flag:
                                    for compen_lidar in compen_lidar_file:
                                        compen_flag = time_stamp_compare(img_file, 1, 2, compen_lidar, 1, 2)
                                        if compen_flag:
                                            shutil.copy(left_lidar, left_new)
                                            shutil.copy(right_lidar, right_new)
                                            shutil.copy(back_lidar, back_new)
                                            shutil.copy(compen_lidar, compen_new)
                                            continue
                                    continue
                            continue
                    continue

    else:
        for i in range(len(img_file_name)):
            name_ls = img_file_name[i].split("_")
            new_str = "pcd" + "_" + name_ls[1] + "_" + name_ls[2][0] + "*"
            optinal_right_data = glob.glob(right_raw + new_str)
            if len(optinal_right_data) == 1:
                flag_right = True
            else:
                flag_right = False

            optinal_left_data = glob.glob(left_raw + new_str)
            if len(optinal_left_data) == 1:
                flag_left = True

            else:
                flag_left = False

            optinal_back_data = glob.glob(back_raw + new_str)
            if len(optinal_back_data) == 1:
                flag_back = True
            else:
                flag_back = False

            optinal_compen_data = glob.glob(compen_raw + new_str)
            if len(optinal_compen_data) == 1:
                flag_compen = True
            else:
                flag_compen = False

            if flag_compen & flag_left & flag_back & flag_right:
                shutil.copy(optinal_left_data[0], left_new)
                shutil.copy(optinal_right_data[0], right_new)
                shutil.copy(optinal_back_data[0], back_new)
                shutil.copy(optinal_compen_data[0], compen_new)
            else:
                print("image %s doesn't have the matched lidar!" % img_file_name[i])
                ignore_img_ls.append(img_file_name[i])
        with open("ignore_img.txt", 'a') as f_imgae:
            s = "\n".join([img for img in ignore_img_ls])
            f_imgae.write(s)


def time_stamp_compare(file_1, s_1, ms_1, file_2, s_2, ms_2):
    file_1_time_stamp = float(file_1.split("_")[s_1] + "." + file_1.split("_")[ms_1][:2])
    file_2_time_stamp = float(file_2.split("_")[s_2] + "." + file_2.split("_")[ms_2][:2])

    if abs(file_1_time_stamp - file_2_time_stamp) < 0.055:
        print(file_1)
        print(file_2)
        print(file_1_time_stamp - file_2_time_stamp)
        return 1
    return 0


def copy_comp_from_img(img_p, raw_comp_p, new_comp_p, compare_with_5ms=True):
    img_file_name = os.listdir(img_p)
    img_file_name.sort()
    if compare_with_5ms:
        raw_pcd_file_name = os.listdir(raw_comp_p)
        raw_pcd_file_name.sort()
        for img_f in img_file_name:
            for pcd_f in raw_pcd_file_name:
                macthed_flag = time_stamp_compare(img_f, 1, 2, pcd_f, 1, 2)
                if macthed_flag:
                    shutil.copy(raw_comp_p+pcd_f, new_comp_p)

    else:
        for i in range(len(img_file_name)):
            name_ls = img_file_name[i].split("_")
            new_str = "pcd" + "_" + name_ls[1] + "_" + name_ls[2][0] + "*"
            optinal_compen_data = glob.glob(raw_comp_p + new_str)
            if len(optinal_compen_data) == 1:
                flag_compen = True
            elif len(optinal_compen_data) == 0:
                flag_compen = False
                print("image %s failed to match with pcd" % img_file_name[i])

                # with open("part5_compen_ignore_id.txt", 'a') as f4:
                #     f4.write(str(new_str) + "\n")
            elif len(optinal_compen_data) == 2:
                flag_compen = False
                print("%s match the same pcd" % img_file_name[i])
            else:
                print("other error!")

            # if flag_compen & flag_left & flag_right & flag_back:
            if flag_compen:
                shutil.copy(optinal_compen_data[0], new_comp_p)



def check_ignored_img(img, points):
    lidar_f_ls = os.listdir(points)
    lidar_f_ls.sort()
    for i in range(len(lidar_f_ls)):
        name_ls = lidar_f_ls[i].strip("png").split("_")
        new_name_str = "image_" + name_ls[1] + "_" + name_ls[2][0] + "*"
        optinal_img = glob.glob(img+new_name_str)
        if len(optinal_img) == 1:
            pass
        elif len(optinal_img) == 0:
            print("points %s unmatched!" % lidar_f_ls[i])
        elif len(optinal_img) == 2:
            print("points %s matched repeated!" % lidar_f_ls[i])
        elif len(optinal_img) == 3:
            print("points %s matched 3 times!" % lidar_f_ls[i])


def copy_img_with_lidar(raw_img, new_img, lidar):
    lidar_ls = os.listdir(lidar)
    lidar_ls.sort()
    additional_img = 0
    for i in range(len(lidar_ls)):
        name_ls = lidar_ls[i].split("_")
        new_name_str = "_".join(["image", name_ls[1], name_ls[2][0]]) + "*"

        optional_img_data = glob.glob(raw_img + new_name_str)
        if len(optional_img_data) == 1:
            shutil.copy(optional_img_data[0], new_img)
        elif len(optional_img_data) == 2:
            additional_img += 1
            shutil.copy(optional_img_data[0], new_img)
        else:
            print("error data!")
            print(optional_img_data)
    print("%d repeated data" % additional_img)



def change_graviti_label_into_4cls(raw_label_path, new_label_path):
    """
    change the label result annotated by graviti into 4 classes: Vehicle, Pedestrian, Cyclist and Unknown
    Cyclist: 'motorcyclist', 'tricycle', 'cyclist'
    Unknown: 'animal', 'barrier', 'bicycle', 'bicycles', 'motorcycle', 'dontcare'
    Pedestrian: 'child', 'adult'
    Vehicle: 'bus', 'car', 'truck'
    """
    for label_f in os.listdir(raw_label_path):
        label_name = raw_label_path + label_f
        with open(label_name, 'r') as f:
            new_rs_ls = []
            for rs_line in f.readlines():
                rs_ls = rs_line.split(" ")
                if rs_ls[0] in ['motorcyclist', 'tricycle', 'cyclist']:
                    rs_ls[0] = 'Cyclist'
                elif rs_ls[0] in ['animal', 'barrier', 'bicycle', 'bicycles', 'motorcycle', 'dontcare']:
                    rs_ls[0] = 'Unknown'
                elif rs_ls[0] in ['child', 'adult']:
                    rs_ls[0] = 'Pedestrian'
                elif rs_ls[0] in ['bus', 'car', 'truck']:
                    rs_ls[0] = 'Vehicle'
                new_rs_ls.append(" ".join(rs_ls))
            new_content = "".join(new_rs_ls)
        with open(new_label_path + label_f, 'w') as f2:
            f2.write(new_content)


def change_graviti_label_into_8cls(raw_label_path, new_label_path):
    """
    change the label result annotated by graviti into 8 classes: Vehicle, Pedestrian, Cyclist, Unknown,
    Tricycle, Bicycles, Barrier, Bicycle

    Cyclist: 'motorcyclist', 'cyclist'
    Tricycle: 'tricycle'
    Unknown: 'animal', 'dontcare'
    Bicycle: 'bicycle', 'motorcycle'
    Pedestrian: 'child', 'adult'
    Vehicle: 'bus', 'car', 'truck'
    Barrier: 'barrier'
    Bicycles: 'bicycles'
    :param raw_label_path:
    :param new_label_path: 
    :return:
    """
    for label_f in os.listdir(raw_label_path):
        label_name = raw_label_path + label_f
        with open(label_name, 'r') as f:
            new_rs_ls = []
            for rs_line in f.readlines():
                rs_ls = rs_line.split(" ")
                if rs_ls[0] in ['motorcyclist', 'cyclist']:
                    rs_ls[0] = 'Cyclist'
                elif rs_ls[0] in ['tricycle']:
                    rs_ls[0] = 'Tricycle'
                elif rs_ls[0] in ['animal', 'dontcare']:
                    rs_ls[0] = 'Unknown'
                elif rs_ls[0] in ['bicycle', 'motorcycle']:
                    rs_ls[0] = 'Bicycle'
                elif rs_ls[0] in ['child', 'adult']:
                    rs_ls[0] = 'Pedestrian'
                elif rs_ls[0] in ['bus', 'car', 'truck']:
                    rs_ls[0] = 'Vehicle'
                elif rs_ls[0] in ['barrier']:
                    rs_ls[0] = 'Barrier'
                elif rs_ls[0] in ['bicycles']:
                    rs_ls[0] = 'Bicycles'
                new_rs_ls.append(" ".join(rs_ls))
            new_content = "".join(new_rs_ls)
        with open(new_label_path + label_f, 'w') as f2:
            f2.write(new_content)


def change_graviti_label_into_6cls(raw_label_path, new_label_path):
    """
    change the label result annotated by graviti into 4 classes: Vehicle, Pedestrian, Cyclist, Unknown, Bicycle_no_person, Bicycles
    """
    for label_f in os.listdir(raw_label_path):
        label_name = raw_label_path + label_f
        with open(label_name, 'r') as f:
            new_rs_ls = []
            for rs_line in f.readlines():
                rs_ls = rs_line.split(" ")
                if rs_ls[0] in ['motorcyclist', 'tricycle', 'cyclist']:
                    rs_ls[0] = 'Cyclist'
                elif rs_ls[0] in ['animal', 'barrier', 'dontcare']:
                    rs_ls[0] = 'Unknown'
                elif rs_ls[0] in ['child', 'adult']:
                    rs_ls[0] = 'Pedestrian'
                elif rs_ls[0] in ['bus', 'car', 'truck']:
                    rs_ls[0] = 'Vehicle'
                elif rs_ls[0] in ['bicycle', 'motorcycle']:
                    rs_ls[0] = 'Bicycle_no_person'
                elif rs_ls[0] in ['bicycles']:
                    rs_ls[0] = 'Bicycles'
                new_rs_ls.append(" ".join(rs_ls))
            new_content = "".join(new_rs_ls)
        with open(new_label_path + label_f, 'w') as f2:
            f2.write(new_content)


# def change_graviti_label_into_6cls(raw_label_path, new_label_path):
#     """
#     change the label result annotated by graviti into 7 classes: Vehicle, Pedestrian, Cyclist, Unknown,
#     Tricycle, Bicycles and Barrier
#     """
#     for label_f in os.listdir(raw_label_path):
#         label_name = raw_label_path + label_f
#         with open(label_name, 'r') as f:
#             new_rs_ls = []
#             for rs_line in f.readlines():
#                 rs_ls = rs_line.split(" ")
#                 if rs_ls[0] in ['motorcyclist', 'motorcycle', 'cyclist', 'bicycle']:
#                     rs_ls[0] = 'Cyclist'
#                 elif rs_ls[0] in ['dontcare']:
#                     rs_ls[0] = 'Unknown'
#                 elif rs_ls[0] in ['child', 'adult']:
#                     rs_ls[0] = 'Pedestrian'
#                 elif rs_ls[0] in ['bus', 'car', 'truck']:
#                     rs_ls[0] = 'Vehicle'
#                 elif rs_ls[0] in ['tricycle']:
#                     rs_ls[0] = 'Tricycle'
#                 elif rs_ls[0] in ['bicycles']:
#                     rs_ls[0] = 'Bicycles'
#                 elif rs_ls[0] in ['barrier', 'animal']:
#                     rs_ls[0] = 'Barrier'
#                 else:
#                     print("error class %s" % rs_ls[0])
#                 new_rs_ls.append(" ".join(rs_ls))
#             new_content = "".join(new_rs_ls)
#         with open(new_label_path + label_f, 'w') as f2:
#             f2.write(new_content)

def change_chaoyang_park_label_into_4cls(raw_label_path, new_label_path):
    """
    change the label result annotated by graviti into 4 classes: Car, Pedestrian, Cyclist and Unknown
    """
    for label_f in os.listdir(raw_label_path):
        label_name = raw_label_path + label_f
        with open(label_name, 'r') as f:
            new_rs_ls = []
            for rs_line in f.readlines():
                rs_ls = rs_line.split(" ")
                # rs_ls[13] = str(float(rs_ls[13]) + float(rs_ls[8])/2)
                # rs_ls[14] = str(-np.pi/2-float(rs_ls[14].strip())) + '\n'
                if rs_ls[0] in ['Cyclist']:
                    rs_ls[0] = 'Cyclist'
                elif rs_ls[0] in ['Others_stationary', 'Non_vehicle', 'Others', 'Others_moving']:
                    rs_ls[0] = 'Unknown'
                elif rs_ls[0] in ['Pedestrian']:
                    rs_ls[0] = 'Pedestrian'
                elif rs_ls[0] in ['Car', 'Vehicle']:
                    rs_ls[0] = 'Vehicle'
                new_rs_ls.append(" ".join(rs_ls))
            new_content = "".join(new_rs_ls)
        with open(new_label_path + label_f, 'w') as f2:


            f2.write(new_content)


def translate_imulabel_into_lidarlabel(imu_label_path, lidar_label_path):
    imu_label_ls = os.listdir(imu_label_path)
    imu_label_ls.sort()

    for imu_label in imu_label_ls:
        lidar_label_file = lidar_label_path + imu_label
        new_lidar_lines = []
        with open(imu_label_path + imu_label, 'r') as f_imu:
            imu_lines = f_imu.readlines()
            for imu_line in imu_lines:
                imu_attris = imu_line.strip().split(" ")
                lidar_attris = copy.deepcopy(imu_attris)
                lidar_attris[11] = str(float(imu_attris[12]))
                lidar_attris[12] = str(-float(imu_attris[11]))
                lidar_attris[13] = str(float(imu_attris[13]) - 1.5)
                lidar_attris[14] = str(float(imu_attris[14])+np.pi/2)
                new_lidar_lines.append(" ".join(lidar_attris))
        with open(lidar_label_file, 'w') as f_lidar:
            f_lidar.write("\n".join(new_lidar_lines))


def translate_lidarpoint_into_imupoint(lidar_point_path, imu_point_path):
    lidar_pc_ls = os.listdir(lidar_point_path)
    lidar_pc_ls.sort()
    for lidar_pc in lidar_pc_ls:
        pc = np.fromfile(lidar_point_path + lidar_pc, dtype=np.float32).reshape(-1, 4)
        imu_pc = copy.deepcopy(pc)
        imu_pc[:, 0] = -pc[:, 1]
        imu_pc[:, 1] = pc[:, 0]
        imu_pc[:, 2] = pc[:, 2] + 1.5
        imu_pc.astype(np.float32).tofile(imu_point_path + lidar_pc)


def generate_id_from_txt(txt_path, id_txt_path):
    file_ls = os.listdir(txt_path)
    file_ls.sort()
    id_ls = [f.strip(".txt") for f in file_ls]
    with open(id_txt_path, 'w') as f1:
        f1.write("\n".join(id_ls))


def rename_left_right_txt_name():
    root_path = "/home/shl/WORKING_DIR/shl_record/part4/part4_right_txt_4cls/"
    file_ls = os.listdir(root_path)
    for f in file_ls:
        raw_name = root_path + f
        new_name = root_path + "%06d.txt" % (int(f.strip(".txt")) + 200000)
        os.rename(raw_name, new_name)


def stat_label(l_path):
    print(l_path)
    classes = set()
    bbox_num = 0
    for l_txt in os.listdir(l_path):
        with open(l_path+l_txt, 'r') as f:
            lines = f.readlines()
            bbox_num += len(lines)
        for line in lines:
            line_ls = line.split(" ")
            classes.add(line_ls[0])
            if line_ls[0] not in ['bus', 'dontcare', 'barrier', 'animal', 'bicycles', 'cyclist', 'car', 'child', 'adult',
                               'truck', 'motorcycle', 'bicycle', 'motorcyclist', 'tricycle']:
                print(l_txt)
    print("There are %d classes" % len(classes), classes)
    print("There are %d 3D bounding boxes" % bbox_num)



def stat_bins(bin_path):
    bin_ls = os.listdir(bin_path)
    points_num = []
    for b_f in bin_ls:
        p_num = np.fromfile(bin_path + b_f, dtype=np.float32).reshape(-1, 4).shape[0]
        points_num.append(p_num)
    print(points_num)
    avg_points = np.mean(np.array(points_num))
    print("avg_points", avg_points)



if __name__ == '__main__':
    fire.Fire()






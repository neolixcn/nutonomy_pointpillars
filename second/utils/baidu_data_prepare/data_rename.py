import os
from shutil import copyfile
import zipfile
camera_dic = {
    "front_3mm":"./images/obstacle",
    "left_3mm":"./images/spherical-left-forward",
    "right_3mm":"./images/spherical-right-forward",
              }

lidar_target_path_name = "velodyne_points"

def time_stamp_compare(file_1, s_1, ms_1, file_2, s_2, ms_2):
    file_1_time_stamp = float(file_1.split("_")[s_1] + "." + file_1.split("_")[ms_1][:2])
    file_2_time_stamp = float(file_2.split("_")[s_2] + "." + file_2.split("_")[ms_2][:2])

    if abs(file_1_time_stamp - file_2_time_stamp) < 0.055:
        print(file_1)
        print(file_2)
        print(file_1_time_stamp - file_2_time_stamp)
        return 1
    return 0

# front_camera_name = "./front_3mm"
# left_camera_name = "./left_3mm"
# right_camera_name = "./right_3mm"
# back_camera_name = "./back_3mm"
# lidar_name = "./veoldy_16"
# lidar_path = "./lidar"
# front_camera_path = "./front"
# left_camera_path = "./left"
# right_camera_path = "./right"
# back_camera_path = "./back"
#
# index = 1
# lidar_dic = {}
# lidar_list = []
# front_list = []
# left_list = []
# time_stamp_list = []
#
# if not (os.path.exists(lidar_name)):
#     os.mkdir(lidar_name)
# camera_list = [(front_camera_path, front_camera_name),
#                (left_camera_path, left_camera_name),
#                (right_camera_path, right_camera_name)]

def get_main_sensor_data(data_path):
    sensor_dic = {}
    for file in os.listdir(data_path):
        file_time_stamp = float(file.split("_")[1] + "." + file.split("_")[2][:2])
        sensor_dic[file_time_stamp] = file
    return sensor_dic

def set_all_sensor_data(sensor_dic,dir_source,dir_target,camera_dic,param_path):

    for key in sorted(sensor_dic):
        file = sensor_dic[key]
        file_time_stamp = key
        copyfile((dir_source + "/" + file), (dir_target + "/" + + file))
        # time_stamp_list.append(file_time_stamp)
        z = zipfile.ZipFile(str(file_time_stamp)+'.zip', 'w')
        z.write(dir_target + "/" + + file)
        z.write(param_path)
        for camera_source,camera_target in camera_dic.items():
            input_dir = camera_source
            output_dir = camera_target
            if not (os.path.exists(output_dir)):
                os.mkdir(output_dir)

            for input_file in os.listdir(input_dir):
                flag = time_stamp_compare(file, input_file)
                if flag:
                    copyfile((input_dir + "/" + input_file),
                             (output_dir + "/" + input_file))
                    z.write(output_dir + "/" + input_file)
                    break

        # index += 1
    return

# root = "."
# from_dir = "../../baidu_data"
# for i in range(1, 21):
#     os.makedirs(root + "/" + str(i) + "/images/front_3mm")
#     #     front_from_dir =
#     os.makedirs(root + "/" + str(i) + "/images/left_3mm")
#     os.makedirs(root + "/" + str(i) + "/images/right_3mm")
#     os.makedirs(root + "/" + str(i) + "/param")
#     os.makedirs(root + "/" + str(i) + "/velodyne_16")
#     front_from_dir = (from_dir + "/images/front_3mm/image_" + str(i).rjust(5, "0") + ".png")
#     front_to_dir = root + "/" + str(i) + "/images/obstacle/image.png"
#     left_from_dir = (from_dir + "/images/left_3mm/image_" + str(i).rjust(5, "0") + ".png")
#     left_to_dir = root + "/" + str(i) + "/images/spherical-left/image.png"
#     right_from_dir = (from_dir + "/images/right_3mm/image_" + str(i).rjust(5, "0") + ".png")
#     right_to_dir = root + "/" + str(i) + "/images/spherical-right/image.png"
#     copyfile(front_from_dir, front_to_dir)
#     copyfile(left_from_dir, left_to_dir)
#     copyfile(right_from_dir, right_to_dir)
#
#     lidar_from = (from_dir + "/velodyne_16/" + "lidar" + str(i).rjust(5, "0") + ".pcd")
#     lidar_to = root + "/" + str(i) + "/velodyne_points/pcd" + str(i).rjust(5, "0") + ".pcd"
#     copyfile(lidar_from, lidar_to)
#
#     param_from = (from_dir + "/param/param.txt")
#     param_to = root + "/" + str(i) + "/param/param.txt"
#     copyfile(param_from, param_to)
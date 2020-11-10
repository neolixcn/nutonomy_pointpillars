import os
import argparse
from shutil import copyfile
parser = argparse.ArgumentParser()
parser.add_argument('--input_lidar',type=str,help='input_file_path lidar')
parser.add_argument('--output_lidar', type=str,help='output_file_path lidar')
parser.add_argument('--input_camera',type=str,help='input_file_path camera')
parser.add_argument('--output_camera', type=str,help='output_file_path camera')
parser.add_argument('--sample_rate', type=int, help='sample rate')
args = parser.parse_args()
def createDir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def char2float(str1, str2):
    in_p = float(str1)
    in_f = float(str2)/(10**(len(str2)))

    return in_p + in_f


def name2float(file):
    name = file.split(".")[0]
    stamp = name.split("_")
    return char2float(stamp[1], stamp[2])

def choose_first(start_time, camera_time):
    for idx, time in enumerate(camera_time):
        if time > start_time:
            return 0
        if camera_time[idx] < start_time and camera_time[idx+1] > start_time:
            if(abs(camera_time[idx] - start_time) < abs(camera_time[idx + 1] - start_time)):
                return idx
            else:
                return idx + 1

def choose_nearest(start_time, camera_time):
    for idx, time in enumerate(camera_time):
        if camera_time[idx] < start_time and camera_time[idx+1] > start_time:
            if(abs(camera_time[idx] - start_time) < abs(camera_time[idx + 1] - start_time)):
                return idx
            else:
                return idx + 1

def sample(ins, out, in_ca, out_ca, rate):
    lidar_files = os.listdir(ins)
    lidar_files.sort()
    # for fi in lidar_files:
    #     print(fi)
    if len(lidar_files) == 0:
        return 
    createDir(out)
    idx = 0
    time_stamps = []
    out_lidar = []
    for file in lidar_files:
        if idx % rate == 0:
            file_name = os.path.join(ins, file)
            copyfile(file_name, os.path.join(out, file))
            out_lidar.append(os.path.join(out, file))
            stamp = name2float(file)
            time_stamps.append(stamp)
        idx += 1
    createDir(out_ca)
    camera_files = os.listdir(in_ca)
    camera_idx = range(len(camera_files))
    camera_times = []
    camera_files.sort()
    for file in camera_files:
        camera_times.append(name2float(file))

    for i, time in enumerate(time_stamps):
        if i == 0:
            idx = choose_first(time, camera_times)
        else:
            idx = choose_nearest(time, camera_times)
        time_str_list = str(time).split(".")
        time_str = time_str_list[0] + "_" + time_str_list[1]
        file_name = "image_" + time_str + ".png"
        #from IPython import embed
        #embed()
        copyfile(os.path.join(in_ca, camera_files[idx]), os.path.join(out_ca, file_name))
	#print()
        

    return True

if __name__ == "__main__":
    inputs_lidar = args.input_lidar
    outputs_lidar = args.output_lidar
    inputs_camera = args.input_camera
    outputs_camera = args.output_camera

    rate = args.sample_rate
    sample(inputs_lidar, outputs_lidar, inputs_camera, outputs_camera, rate)




#!/usr/bin/env python3

###############################################################################
# Copyright 2018 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

"""
function to parse lidar data from *.record files, created using Apollo-Auto
parsed data is saved to *.txt file, for each scan
current implementation for:
* Velodyne VLS-128 lidar
"""

import os
import sys

from cyber_py3 import cyber
from cyber_py3 import record
from sensor_pointcloud_pb2 import PointCloud
HEADER = '''\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4 
TYPE F F F F 
COUNT 1 1 1 1 
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii
'''

def write_pcd(points, save_pcd_path):
    n = len(points)
    lines = []
    for i in range(n):
        x, y, z, i, is_g = points[i]
        lines.append('{:.6f} {:.6f} {:.6f} {}'.format( \
            x, y, z, i))
    with open(save_pcd_path, 'w') as f:
        f.write(HEADER.format(n, n))
        f.write('\n'.join(lines))


def parse_data(channelname, msg, out_folder):
    """
    """
    msg_lidar = PointCloud()
    msg_lidar.ParseFromString(msg)
    # from IPython import embed
    # embed()
    nPts = len(msg_lidar.point)
    #from IPython import embed
    #embed()
    pcd = []
    for j in range(nPts):
        p = msg_lidar.point[j]
        pcd.append('{:.6f} {:.6f} {:.6f} {}'.format(p.x, p.y, p.z, p.intensity))
    # for j in range(nPts):
    #    p = msg_lidar.point[j]
    #    if (p.x >=0 )or(p.x < 0):
    #     pcd.append([p.x, p.y, p.z, p.intensity])

    tstamp = msg_lidar.measurement_time
    temp_time = str(tstamp).split('.')

    if len(temp_time[1]) == 1:
        temp_time1_adj = temp_time[1] + '0'
    else:
        temp_time1_adj = temp_time[1]

    pcd_time = temp_time[0] + '_' + temp_time1_adj
    pcd_filename = "pcd_" + pcd_time + ".pcd"
    with open(out_folder + pcd_filename, 'w') as f:
        f.write(HEADER.format(nPts, nPts))
        f.write('\n'.join(pcd))
    """
    with open(out_folder + pcd_filename, 'w') as outfile:
        # print("point cloud is writen to " + out_folder + pcd_filename)
        for item in pcd:
            data = str(item)[1:-1]
            outfile.write("%s\n" % data)
    """
    return tstamp

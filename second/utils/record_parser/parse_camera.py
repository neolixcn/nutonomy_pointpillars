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
function to parse camera images from *.record files, created using Apollo-Auto
parsed data is saved to *.jpeg file, for each capture
"""

import os
import sys

from cyber_py3 import cyber
from cyber_py3 import record
from sensor_image_pb2 import Image
import cv2
import numpy as np


def parse_data(channelname, msg, out_folder):
    """
    parser images from Apollo record file
    """
    msg_camera = Image()
    # from IPython import embed
    # embed()
    msg_camera.ParseFromString(msg)
    # from IPython import embed
    # embed()
    tstamp = msg_camera.header.timestamp_sec

    temp_time = str(tstamp).split('.')
    if len(temp_time[1]) == 1:
        temp_time1_adj = temp_time[1] + '0'
    else:
        temp_time1_adj = temp_time[1]
    image_time = temp_time[0] + '_' + temp_time1_adj
    # from IPython import embed
    # embed()
    img = np.frombuffer(msg_camera.data, dtype=np.uint8) 
    image = img.reshape(1024, 1280, 3)
    #image = img.reshape(1080, 1920, 3)
    bgr_img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_folder + "image_" + image_time + ".png", bgr_img)
    # image_filename = "image_" + image_time + ".jpeg"
    # print(out_folder + image_filename)
    # f = open(out_folder + image_filename, 'w+b')
    # f.write(msg_camera.data)
    # f.close()
    
    
    
    return tstamp

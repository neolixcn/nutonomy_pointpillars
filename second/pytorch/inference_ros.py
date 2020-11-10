# -*- coding: utf-8 -*-
import os
import argparse
import pathlib
import pickle
import shutil
import time
from functools import partial
import sys
sys.path.append('../')
from pathlib import Path
import fire
import numpy as np
import torch
import torch.nn as nn
import os
print(torch.__version__)
print(os.environ['PYTHONPATH'])
from google.protobuf import text_format

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar


def get_paddings_indicator(actual_num, max_num, axis=0):
    """
    Create boolean mask by actually number of a padded tensor.
    :param actual_num:
    :param max_num:
    :param axis:
    :return: [type]: [description]
    """
    actual_num = torch.unsqueeze(actual_num, axis+1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis+1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num : [N, M, 1]
    # tiled_actual_num : [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # title_max_num : [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape : [batch_size, max_num]
    return paddings_indicator

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted


def example_convert_to_torch(example, dtype=torch.float32, device=None) -> dict:
    # device = device or torch.device("cuda:0")
    example_torch = {}
    # float_names = ["voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect", "Trv2c", "P2"]
    float_names = ["voxels", "anchors", "reg_targets", "reg_weights", "bev_map"]


    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype).cuda()
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(v, dtype=torch.int32).cuda()
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(v, dtype=torch.uint8).cuda()
            # torch.uint8 is now deprecated, please use a dtype torch.bool instead
        else:
            example_torch[k] = v
    return example_torch


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


def publish_test(np_p_ranged, frame_id):
    header = Header()
    header.stamp = rospy.Time()
    header.frame_id = frame_id

    x = np_p_ranged[:, 0].reshape(-1)
    y = np_p_ranged[:, 1].reshape(-1)
    z = np_p_ranged[:, 2].reshape(-1)

    if np_p_ranged.shape[1] == 4:
        i = np_p_ranged[:, 3].reshape(-1)
    else:
        i = np.zeros(np_p_ranged.shape[0], 1).reshape(-1)
    cloud = np.stack((x, y, z, i))
    msg_segment = pc2.create_cloud(header=header,
                                   fields=_make_point_field(4),
                                   points=cloud.T)
    pub_points.publish(msg_segment)



def predict_kitti_to_anno(net,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):

    # eval example : [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
    #                 4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
    #                 8: 'image_idx', 9: 'image_shape']

    # eval example [0: 'voxels', 1: 'num_points', 2: 'coordinate', 3: 'anchors',
    # 4: 'anchor_mask', 5: 'pc_idx']

    pillar_x = example[0][:, :, 0].unsqueeze(0).unsqueeze(0)
    pillar_y = example[0][:, :, 1].unsqueeze(0).unsqueeze(0)
    pillar_z = example[0][:, :, 2].unsqueeze(0).unsqueeze(0)
    pillar_i = example[0][:, :, 3].unsqueeze(0).unsqueeze(0)
    num_points_per_pillar = example[1].float().unsqueeze(0)

    # Find distance of x, y, and z from pillar center
    # assuming xyres_16.proto
    coors_x = example[2][:, 3].float()
    coors_y = example[2][:, 2].float()
    x_sub = coors_x.unsqueeze(1) * 0.16 -22.96 #+ 0.08#-22.96#+ 0.08#-22.96#-19.76
    y_sub = coors_y.unsqueeze(1) * 0.16 -22.96#- 19.76 #-22.96#-19.76#-22.96#-19.76
    ones = torch.ones([1, 100], dtype=torch.float32, device=pillar_x.device)
    x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
    y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

    num_points_for_a_pillar = pillar_x.size()[3]
    mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
    mask = mask.permute(0, 2, 1)
    mask = mask.unsqueeze(1)
    mask = mask.type_as(pillar_x)

    coors = example[2]
    anchors = example[3]
    anchors_mask = example[4]
    anchors_mask = torch.as_tensor(anchors_mask, dtype=torch.uint8, device=pillar_x.device)
    anchors_mask = anchors_mask.byte()
    # rect = example[3]
    # Trv2c = example[4]
    # P2 = example[5]
    pc_idx = example[5]

    input = [pillar_x, pillar_y, pillar_z, pillar_i,
             num_points_per_pillar, x_sub_shaped, y_sub_shaped,
             mask, coors, anchors, anchors_mask, pc_idx]

    predictions_dicts = net(input)
    # lidar_box, final_score, label_preds, pc_idx

    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        # image_shape = batch_image_shape[i]
        pc_idx = preds_dict[3]

        if preds_dict[0] is not None: # bbox list
            # box_2d_preds = preds_dict[0].detach().cpu().numpy() # bbox
            # box_preds = preds_dict[1].detach().cpu().numpy() # bbox3d_camera
            scores = preds_dict[1].detach().cpu().numpy() # scores
            box_preds_lidar = preds_dict[0].detach().cpu().numpy() # box3d_lidar
            # write pred to file
            label_preds = preds_dict[2].detach().cpu().numpy() # label_preds

            anno = kitti.get_start_result_anno()
            num_example = 0
            content = ''
            for box_lidar, score, label in zip(
                    box_preds_lidar, scores, label_preds):
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                content += str(label) + " 0.0 0 0.0 0.0 0.0 0.0 0.0 " + str(box_lidar[5]) + " " + str(box_lidar[3]) + " "\
                           + str(box_lidar[4]) + " " + str(box_lidar[0]) + " " + str(box_lidar[1]) + " " + str(box_lidar[2]) + " " + str(box_lidar[6]) + " " + str(score) + "\n"
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box_lidar[6])
                anno["bbox"].append(np.array([0, 0, 0, 0]))
                anno["dimensions"].append([box_lidar[4], box_lidar[5], box_lidar[3]]) # annotate by shl
                # anno["dimensions"].append(box_lidar[3:6])
                anno["location"].append(box_lidar[:3])
                anno["rotation_y"].append(box_lidar[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            content = content.strip()


def delete_nan_points(points):
        new_poins = np.array([])
        print(points)
        for i in range(points.shape[0]):
            if (np.isnan(points[i][0])) | (np.isnan(points[i][1])) | (np.isnan(points[i][2])):
                pass
            else:
                np.row_stack((new_poins, points[i]))
        return new_poins


def callback(msg):
    arr_bbox = BoundingBoxArray()

    # pcl_msg = pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z", "intensity", "ring"))
    pcl_msg = pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z"))
    # pcl_msg = pc2.read_points(msg, skip_nans=True)
    # print(pcl_msg)
    np_p = np.array(list(pcl_msg), dtype=np.float32)
    np_p = np.column_stack((np_p, np.zeros((np_p.shape[0], 1))))
    print(np_p)
    # np_p = np.delete(np_p, -1, 1)  # delete "ring" field
    # np_p = delete_nan_points(np_p)


    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        inference=True,
        points=np_p)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=input_cfg.batch_size,
        shuffle=False,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    net.eval()
    global_set = None
    eval_data = iter(eval_dataloader)
    example = next(eval_data)
    example = example_convert_to_torch(example, torch.float32)
    example_tuple = list(example.values())
    example_tuple[5] = torch.from_numpy(example_tuple[5])

    result = predict_kitti_to_anno(net, example_tuple, class_names, center_limit_range, model_cfg.lidar_input, global_set)
    print("result", result)






# def evaluate(config_path,
#              model_dir,
#              result_path=None,
#              predict_test=False,
#              ckpt_path=None,
#              ref_detfile=None,
#              pickle_result=True,
#              read_predict_pkl_path=None):
#
#     model_dir = str(Path(model_dir).resolve())
#     if predict_test:
#         result_name = 'predict_test'
#     else:
#         result_name = 'eval_results'
#     if result_path is None:
#         model_dir = Path(model_dir)
#         result_path = model_dir / result_name
#     else:
#         result_path = pathlib.Path(result_path)
#
#     if isinstance(config_path, str):
#         config = pipeline_pb2.TrainEvalPipelineConfig()
#         with open(config_path, "r") as f:
#             proto_str = f.read()
#             text_format.Merge(proto_str, config)
#     else:
#         config = config_path
#
#     input_cfg = config.eval_input_reader
#     model_cfg = config.model.second
#     train_cfg = config.train_config
#     class_names = list(input_cfg.class_names)
#     center_limit_range = model_cfg.post_center_limit_range
#     #########################
#     # Build Voxel Generator
#     #########################
#     voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
#     bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
#     box_coder = box_coder_builder.build(model_cfg.box_coder)
#     target_assigner_cfg = model_cfg.target_assigner
#     target_assigner = target_assigner_builder.build(target_assigner_cfg,
#                                                     bv_range, box_coder)
#
#     net = second_builder.build(model_cfg, voxel_generator, target_assigner, input_cfg.batch_size)
#     net.cuda()
#     if train_cfg.enable_mixed_precision:
#         net.half()
#         net.metrics_to_float()
#         net.convert_norm_to_float(net)
#
#     if ckpt_path is None:
#         torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
#     else:
#         torchplus.train.restore(ckpt_path, net)
#
#     eval_dataset = input_reader_builder.build(
#         input_cfg,
#         model_cfg,
#         training=False,
#         voxel_generator=voxel_generator,
#         target_assigner=target_assigner)
#
#     eval_dataloader = torch.utils.data.DataLoader(
#         eval_dataset,
#         batch_size=input_cfg.batch_size,
#         shuffle=False,
#         num_workers=input_cfg.num_workers,
#         pin_memory=False,
#         collate_fn=merge_second_batch)
#
#     if train_cfg.enable_mixed_precision:
#         float_dtype = torch.float16
#     else:
#         float_dtype = torch.float32
#
#     net.eval()
#     result_path_step = result_path / f"step_{net.get_global_step()}"
#     result_path_step.mkdir(parents=True, exist_ok=True)
#     t = time.time()
#     dt_annos = []
#     global_set = None
#     eval_data = iter(eval_dataloader)
#     example = next(eval_data)
#     example = example_convert_to_torch(example, float_dtype)
#     example_tuple = list(example.values())
#     example_tuple[5] = torch.from_numpy(example_tuple[5])
#     if (example_tuple[3].size()[0] != input_cfg.batch_size):
#         continue
#
#     dt_annos += predict_kitti_to_anno(
#         net, example_tuple, class_names, center_limit_range,
#         model_cfg.lidar_input, global_set)

    # for example in iter(eval_dataloader):
    #     # eval example [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
    #     #               4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
    #     #               8: 'image_idx', 9: 'image_shape']
    #
    #     # eval example [0: 'voxels', 1: 'num_points', 2: 'coordinate', 3: 'anchors',
    #     # 4: 'anchor_mask', 5: 'pc_idx']
    #     example = example_convert_to_torch(example, float_dtype)
    #     # eval example [0: 'voxels', 1: 'num_points', 2: 'coordinate', 3: 'anchors',
    #     # 4: 'anchor_mask', 5: 'pc_idx']
    #
    #     example_tuple = list(example.values())
    #     example_tuple[5] = torch.from_numpy(example_tuple[5])
    #     # example_tuple[9] = torch.from_numpy(example_tuple[9])
    #
    #     if (example_tuple[3].size()[0] != input_cfg.batch_size):
    #         continue
    #
    #     dt_annos += predict_kitti_to_anno(
    #         net, example_tuple, class_names, center_limit_range,
    #         model_cfg.lidar_input, global_set)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing')
    args = parser.parse_args()

    model_dir = "/nfs/nas/model/songhongli/neolix_shanghai_3828/"
    config_path = "/home/songhongli/Projects/pointpillars2/second/configs/pointpillars/xyres_16_4cls.proto"
    if isinstance(config_path, str):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range
    #########################
    # Build Voxel Generator
    #########################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

    net = second_builder.build(model_cfg, voxel_generator, target_assigner, input_cfg.batch_size)
    net.cuda()
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])

    #  code added for using ROS
    rospy.init_node('pointpillars_ros_node')

    sub_ = rospy.Subscriber("/sensor/velodyne16/all/compensator/PointCloud2", PointCloud2, callback, queue_size=1)

    pub_points = rospy.Publisher("points_modified", PointCloud2, queue_size=1)
    pub_arr_bbox = rospy.Publisher("pre_arr_bbox", BoundingBoxArray, queue_size=10)
    # pub_bbox = rospy.Publisher("voxelnet_bbox", BoundingBox, queue_size=1)
    print("[+] voxelnet_ros_node has started!")
    rospy.spin()




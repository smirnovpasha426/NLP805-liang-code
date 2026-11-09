import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
import math
from tqdm import tqdm
import cv2
import numpy as np
import math
import json
import glob
import argparse
import pickle as pkl
import multiprocessing as mp
from read_write_models import read_model, write_model
from scipy.spatial.transform import Rotation as R

from unidepth.utils import colorize, image_grid
from unidepth.models import UniDepthV1

def quaternion_to_euler(qw, qx, qy, qz):
    r = R.from_quat([qx, qy, qz, qw])
    return r.as_euler('xyz', degrees=True)  # Returns roll, pitch, yaw in degrees

def quaternion_to_rotation_matrix(quaternion):
    """Converts a quaternion to a 3x3 rotation matrix."""
    qw, qx, qy, qz = quaternion
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def calculate_camera_world_position(quaternion, translation):
    """Calculates the camera's world position from the given quaternion and translation vector."""
    # Convert the quaternion to a 3x3 rotation matrix
    R = quaternion_to_rotation_matrix(quaternion)

    # Compute the transpose of the rotation matrix
    R_inv = R.T

    # Compute the camera center in world coordinates
    translation_vector = np.array(translation)
    camera_world_position = -np.dot(R_inv, translation_vector)

    return camera_world_position

def parse_images_bin(file_path):
    cameras, images, points3D = read_model(file_path, ext=".bin")

    images_data = {}
    for image_id, image in images.items():
        qw, qx, qy, qz = image.qvec
        quaternion = (qw, qx, qy, qz)
        tx, ty, tz = image.tvec
        translation = (tx, ty, tz)
        camera_id = image.camera_id
        image_name = image.name
        roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz)
        images_data[image_id] ={
            'quaternion': (qw, qx, qy, qz), 
            'translation': (tx, ty, tz),
            'camera_world_position': calculate_camera_world_position(quaternion, translation),
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'camera_id': camera_id,
            'image_name': image_name,
            'frame_id': int(image_name[-8:-4]),
            'xys': image.xys,#每一个3d点在图像中的投影点
            'point3D_ids': image.point3D_ids#每一个3d点在图像中的id
        }

    points3D_data = {}
    for point3D_id, point3D in points3D.items():
        points3D_data[point3D_id] = {
            'xyz': point3D.xyz,
            'rgb': point3D.rgb,
            'error': point3D.error,
            'image_ids': point3D.image_ids,#每一个3d点在哪些图像中出现
            'point2D_idxs': point3D.point2D_idxs#每一个3d点在每一张图像中的索引
        }

    cameras_data = {}
    for camera_id, camera in cameras.items():
        cameras_data[camera_id] = {
            'model': camera.model,
            'width': camera.width,
            'height': camera.height,
            'params': camera.params
        }
    return cameras_data, images_data, points3D_data

#撰写函数，过滤error大于全部点云error排序前百分之一的点，然后选出这其中point2D_idxs的长度大于2的点
def filter_points(points3D):
    error_list = []
    for point3D_id, point3D in points3D.items():
        error_list.append(point3D["error"])
    # error_mean = np.mean(error_list)
    filter_points3D = {}
    # for point3D_id, point3D in points3D.items():
    #     if point3D["error"] < error_mean * 0.05 and len(point3D["point2D_idxs"]) > 2:
    #         filter_points3D[point3D_id] = point3D
    #我希望你实现一个逻辑是保留error排在前百分之一的点，然后再选出这些点中point2D_idxs的长度大于2的点
    error_list.sort()
    error_list = error_list[:int(len(error_list) * 0.01)]
    for point3D_id, point3D in points3D.items():
        if point3D["error"] in error_list and len(point3D["point2D_idxs"]) > 2:
            filter_points3D[point3D_id] = point3D
    return filter_points3D


def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)#弧度
        return angle
    return 0

#撰写函数，计算filter_points中每一个3d点所对应的2d图像互相之间的最大视差角（Parallax），并返回最大视差角最大的那个3d点的id
def compute_max_parallax(filter_points3D, images_data):
    max_parallax = 0
    max_parallax_point3D_id = 0
    for point3D_id, point3D in filter_points3D.items():
        image_ids = point3D["image_ids"]

        point3D_xyz = filter_points3D[point3D_id]["xyz"]

        image_num = len(image_ids)
        for i in range(image_num):
            image_id_i = int(image_ids[i])
            if image_id_i not in images_data :
                    continue
            for j in range(i+1, image_num):
                image_id_j = int(image_ids[j])
                if image_id_j not in images_data:
                    continue
                # roll_i, pitch_i, yaw_i = images_data[image_id_i]['roll'], images_data[image_id_i]['pitch'], images_data[image_id_i]['yaw']
                # roll_j, pitch_j, yaw_j = images_data[image_id_j]['roll'], images_data[image_id_j]['pitch'], images_data[image_id_j]['yaw']
                # parallax = math.sqrt((roll_i - roll_j)**2 + (pitch_i - pitch_j)**2 + (yaw_i - yaw_j)**2)
                parallax = dot_product_angle((np.array(images_data[image_id_i]['camera_world_position']) - point3D_xyz) , (np.array(images_data[image_id_j]['camera_world_position']) - point3D_xyz))
                if parallax > max_parallax:
                    max_parallax = parallax
                    max_parallax_point3D_id = point3D_id
    return max_parallax_point3D_id


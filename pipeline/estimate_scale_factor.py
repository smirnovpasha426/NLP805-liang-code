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


#撰写函数，接受一个3d点的id，计算这个3d点在所有图像中的投影点，然后根据视差角筛选出视差角大于均值的图像对，最后返回这些图像对的id
def compute_parallax_by_projection(point3D_id, filter_points3D, images_data):
    image_ids = filter_points3D[point3D_id]["image_ids"]
    point3D_xyz = filter_points3D[point3D_id]["xyz"]

    image_num = len(image_ids)
    parallax_dict = {}
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

            parallax_dict[(image_id_i, image_id_j)] = parallax
    parallax_mean = np.mean(list(parallax_dict.values()))
    parallax_ids = []
    # for (image_id_i, image_id_j), parallax in parallax_dict.items():
    #     if parallax > parallax_mean:
    #         parallax_ids.append((image_id_i, image_id_j))
    #修改筛选规则为获取视角差大于均值一半的图像对中的最大的前8组图像对，如果不足8对则都保留
    for (image_id_i, image_id_j), parallax in parallax_dict.items():
        if parallax > parallax_mean * 0.1:
            parallax_ids.append((image_id_i, image_id_j))
    parallax_ids = sorted(parallax_ids, key=lambda x: parallax_dict[x], reverse=True)
    # parallax_ids = parallax_ids[:10]#会不会超过上限？ 会的，所以需要在前面加一个判断
    if len(parallax_ids) > 8:
        parallax_ids = parallax_ids[:8]
    #储存最后parallax_ids中存在的图像对的id到一个列表中
    idin_parallax_ids = []
    for (image_id_i, image_id_j) in parallax_ids:
        idin_parallax_ids.append(image_id_i)
        idin_parallax_ids.append(image_id_j)
    #给parallax_ids去重
    parallax_ids = list(set(parallax_ids))
    return parallax_ids,idin_parallax_ids

#撰写函数，将COLMAP的相机参数转换为相机内参矩阵
def params_to_intrinsics(params):
    """
    Convert camera parameters from COLMAP to the intrinsic matrix.

    Parameters:
    - params (list or numpy array): Camera parameters [focal length, cx, cy, radial distortion]

    Returns:
    - numpy.ndarray: The intrinsic camera matrix.
    """
    # 确保输入是一个长度至少为3的列表或数组
    if len(params) < 4:
        raise ValueError("Parameters must include at least focal length, cx, and cy")
    
    # 创建一个3x3的零矩阵
    intrinsics_matrix = np.zeros((3, 3))
    
    # 设置对角线上的焦距和缩放因子
    intrinsics_matrix[0, 0] = params[0]  # fx
    intrinsics_matrix[1, 1] = params[0]  # fy，假设fx和fy相同
    
    # 设置主点坐标
    intrinsics_matrix[0, 2] = params[1]  # cx
    intrinsics_matrix[1, 2] = params[2]  # cy
    
    # 设置齐次坐标系的缩放因子
    intrinsics_matrix[2, 2] = 1.0
    #将精度转化为float32
    intrinsics_matrix = intrinsics_matrix.astype(np.float32)

    return intrinsics_matrix

#在输入cameras、images和idin_parallax_ids的情况下，预测深度图像，并保存。rgb图像的路径为/home/user/demo-video/0qw2HOHjiyU_whole/imgs_8fps_360p/  图像的名称为output_frame_2600.png，替换2600为idin_parallax_ids中的每一个id 。我需要你提供一个函数，输入cameras和idin_parallax_ids，输出预测的深度图像。注意这个函数要使用camera的参数，不要使用默认的参数。
def predict_depth_image(model, max_parallax_point3D_id, cameras, images, idin_parallax_ids):
    idin_parallax_ids_with_depth = {}
    for image_id in idin_parallax_ids:
        image_id_int = int(image_id)
        image = images[image_id_int]
        camera = cameras[image["camera_id"]]

        #如果image_id不足四位数的话，需要在前面补0，补成四位数字符串
        # image_id_str = str(image_id_int).zfill(4)
        rgb = np.array(Image.open(os.path.join(image_path, image["image_name"])))
        
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1) # C, H, W

        intrinsics_matrix = params_to_intrinsics(camera["params"])
        intrinsics_torch = torch.from_numpy(intrinsics_matrix)
        

        predictions = model.infer(rgb_torch, intrinsics_torch)
        depth_pred = predictions["depth"].squeeze().cpu().numpy()

        # depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")
        viz_depth = colorize(predictions["depth"], vmin=0.01, vmax=10.0,cmap='plasma')
        plt.imshow(viz_depth.cpu().squeeze(), cmap='plasma')
        plt.axis('off')
        plt.show()
        
        #根据max_parallax_point3D_id查询当前image_id对应的3d点的投影点，然后将投影点的坐标转换为像素坐标，最后利用像素坐标，读出深度图像的深度值，将深度值和image_id保存到idin_parallax_ids_with_depth中
        # point3D = filter_points3D[max_parallax_point3D_id]

        idx_point3D = np.where(image["point3D_ids"] == max_parallax_point3D_id)[0][0]
        point2D_posi = image["xys"][idx_point3D]
        # print("point2D_posi",point2D_posi)
        
        # idx = np.where(point3D["image_ids"] == image_id_int)[0][0]
        # point2D_idx = point3D["point2D_idxs"][idx]#找到3d点在当前图像中的索引
        # print("point2D_idx",point3D["point2D_idxs"][idx])
        # print("image_id_int",image_id_int)
        # point2D = image["xys"][point2D_idx]#找到3d点在当前图像中的投影点
        # point3D = np.array([point2D[0], point2D[1], 1])#将3d点的坐标转换为齐次坐标
        # print("point2D",point2D)
        # point2D = np.dot(np.linalg.inv(camera["params"]), point2D)#将3d点的坐标转换为像素坐标
        # point2D = point2D[:2] / point2D[2]#将像素坐标转换为整数

        depth = depth_pred[int(point2D_posi[1]), int(point2D_posi[0])]
        idin_parallax_ids_with_depth[image_id_int] = depth#将深度值和image_id保存到idin_parallax_ids_with_depth中
        
    return idin_parallax_ids_with_depth

def remove_outliers(data, std_factor=2):
    """
    剔除数据中的明显异常值

    参数：
    data (list): 输入数据列表
    std_factor (int, optional): 定义异常值的标准差倍数，默认为3

    返回：
    list: 剔除异常值后的数据列表
    """
    mean = np.mean(data)
    std = np.std(data)
    
    lower_bound = mean - std_factor * std
    upper_bound = mean + std_factor * std
    
    filtered_data_idx = [i for i, x in enumerate(data) if lower_bound < x < upper_bound]
    filtered_data = [data[i] for i in filtered_data_idx]
    
    return filtered_data, filtered_data_idx




    #按照parallax_ids，根据已经计算出的每个视角的深度值，以及相机内外参，利用余弦定理计算image_id_i和image_id_j之间的距离，再除以这两个图像的坐标系下的距离，得到深度图像的尺度因子，并返回均值、最大值和最小值、标准差、中位数
def compute_scale_factor(parallax_ids, idin_parallax_ids_with_depth, images_data, max_parallax_point3D_id, filter_points3D):
    scale_factors = []
    full_record = []
    for image_id_i, image_id_j in parallax_ids:
        image_id_i = int(image_id_i)
        image_id_j = int(image_id_j)
        depth_i = idin_parallax_ids_with_depth[image_id_i]
        depth_j = idin_parallax_ids_with_depth[image_id_j]
        # camera_i = cameras[images_data[image_id_i]['camera_id']]
        # camera_j = cameras[images_data[image_id_j]['camera_id']]
        #计算两个图像视线的夹角
        # roll_i, pitch_i, yaw_i = images_data[image_id_i]['roll'], images_data[image_id_i]['pitch'], images_data[image_id_i]['yaw']
        # roll_j, pitch_j, yaw_j = images_data[image_id_j]['roll'], images_data[image_id_j]['pitch'], images_data[image_id_j]['yaw']
        # angle = math.sqrt((roll_i - roll_j)**2 + (pitch_i - pitch_j)**2 + (yaw_i - yaw_j)**2)
        #角度是3d点发射向两个视角的相机坐标的向量夹角
        point3D_xyz = filter_points3D[max_parallax_point3D_id]["xyz"]
        angle = dot_product_angle((np.array(images_data[image_id_i]['camera_world_position']) - point3D_xyz) , (np.array(images_data[image_id_j]['camera_world_position']) - point3D_xyz))

        #计算两个图像的距离
        distance = np.linalg.norm(np.array(images_data[image_id_i]['camera_world_position']) - np.array(images_data[image_id_j]['camera_world_position']))#计算两个图像的距离 np.linalg.norm()计算向量的范数
        #利用余弦定理计算两图像的真实距离
        cos_angle = math.cos(angle)
        cos_angle = max(-1, min(1, cos_angle))  # 限制 cos_angle 在 [-1, 1] 范围内
        real_distance = math.sqrt(depth_i**2 + depth_j**2 - 2 * depth_i * depth_j * cos_angle)

        #计算尺度因子
        #real_distance是对数尺度的，所以需要取指数
        # real_distance = math.exp(real_distance)
        if real_distance > 9.5 :
            continue
        scale_factor = real_distance / distance
        print(scale_factor,real_distance,distance)
        scale_factors.append(scale_factor)
        full_record.append([scale_factor,real_distance,distance])
    filtered_scale_factors, filtered_scale_factors_idx = remove_outliers(scale_factors)#我发现有时候有异常值，实现一个功能 剔除scale_factors明显的异常值
    # print(filtered_scale_factors)
    if len(filtered_scale_factors) == 0:
        return {
            'mean': 0,
            'max': 0,
            'min': 0,
            'std': 0,
            'median': 0
        }, {}

    

    return {
        'mean': np.mean(filtered_scale_factors),
        'max': np.max(filtered_scale_factors),
        'min': np.min(filtered_scale_factors),
        'std': np.std(filtered_scale_factors),
        'median': np.median(filtered_scale_factors)
    }, 
    {'full_record': full_record,
     'filtered_scale_factors_idx': filtered_scale_factors_idx,
     'parallax_ids': parallax_ids
    }

#实现一个函数，输入用来没间隔1.5米取一个点。参考已经计算出的尺度因子均值，沿着image_id计算其与后续帧的image["translation"]距离，每当其距离乘以尺度因子的结构大于1.5的时候，就储存该点的数据到一个新字典
#所以你要首先排序所有的image_id，然后新建一个sparse_frame字典，先储存最小的image_id到这个字典中，然后按照顺序计算最小的image_id开始，与后面的image_id的之间的image["translation"]距离，然后乘以尺度因子，如果大于1.5米，就储存该点的数据到一个该字典中
def get_sparse_frame(images_data, scale_factors_dict):
    frames_data = {}
    for image_id in list(images_data.keys()):
        frame_id = images_data[image_id]["frame_id"]
        # frames_data["image_id"] = image_id
        if frame_id not in list(frames_data.keys()):
            frames_data[frame_id] = images_data[image_id]
            frames_data[frame_id]["image_id"] = image_id
            
    frames_ids = list(frames_data.keys())
    frames_ids.sort()
    
    sparse_frame = {}
    sparse_frame[frames_ids[0]] = {
                'distance_with_lastframe' : 0.01,
                'image_name': frames_data[frames_ids[0]]["image_name"],
                'image_id': frames_data[frames_ids[0]]["image_id"],
                'frame_id': frames_data[frames_ids[0]]["frame_id"]
            }

    last_index = 0  # Start with the first image
    for i in range(1, len(frames_ids)):
        current_frame_id = frames_ids[i]
        last_frame_id = frames_ids[last_index]
        # Calculate the translation distance between the current and last stored image
        translation_distance = frames_data[current_frame_id]['camera_world_position']
        last_translation_distance = frames_data[last_frame_id]['camera_world_position']


        # distance = ((translation_distance[0] - last_translation_distance[0]) ** 2 +
        #             (translation_distance[1] - last_translation_distance[1]) ** 2 +
        #             (translation_distance[2] - last_translation_distance[2]) ** 2) ** 0.5
        distance = ((translation_distance[0] - last_translation_distance[0]) ** 2 +
                    (translation_distance[2] - last_translation_distance[2]) ** 2 ) ** 0.5

        # Apply the scale factor
        scale_factor = scale_factors_dict['mean']
        scaled_distance = distance * scale_factor

        # Check if the scaled distance is greater than 1 meters
        if scaled_distance > 2.45:
            #把scaled_distance转化为浮点数
            scaled_distance = float(scaled_distance)
            sparse_frame[current_frame_id] = {
                'distance_with_lastframe' : scaled_distance,
                'image_name': frames_data[current_frame_id]["image_name"],
                'image_id': frames_data[current_frame_id]["image_id"],
                'frame_id': frames_data[current_frame_id]["frame_id"]
            }
            last_index = i
    return sparse_frame


#实现一个可视化代码，在接受sparse_frame的情况下，将每六个图像的数据可视化到一张图像中，每一行显示一个图像，图像下方显示这6张图像的sparse_frame[current_image_id]["distance_with_lastframe"]的值
# image_id_str = str(image_id_int).zfill(4)
#         rgb = np.array(Image.open("/home/user/demo-video/0qw2HOHjiyU_whole/imgs_8fps_360p/output_frame_{}.png".format(image_id_str)))
#读取rgb的代码见上
def visualize_sparse_frame(sparse_frame,sence_id):
    sparse_frame_image_ids = list(sparse_frame.keys())
    # image_ids.sort()
    for i in range(0, len(sparse_frame_image_ids), 6):
        fig, axs = plt.subplots(1, 6, figsize=(20, 5))
        for j in range(6):
            if i + j < len(sparse_frame_image_ids):
                frame_id = sparse_frame_image_ids[i + j]
                # rgb = np.array(Image.open(image_path + sparse_frame[frame_id]["image_name"]))
                rgb = np.array(Image.open(os.path.join(image_path, sparse_frame[frame_id]["image_name"])))
                axs[j].imshow(rgb)
                # axs[j].set_title(f"Image ID: {image_id}  Frame ID: {sparse_frame[image_id]['frame_id']}")
                axs[j].set_title(f" F: {frame_id}   I: {sparse_frame[frame_id]['image_id']}")
                if "distance_with_lastframe" in sparse_frame[frame_id]:
                    axs[j].set_xlabel(f"Distance: {sparse_frame[frame_id]['distance_with_lastframe']:.2f}")
        plt.show()
        #新建文件夹储存以上可视化例子
        os.makedirs(f"25new_100_wo_merge_sparse_frame/{sence_id}", exist_ok=True)
        plt.savefig(f"25new_100_wo_merge_sparse_frame/{sence_id}/{i}.png")
        plt.close()
    return

parser = argparse.ArgumentParser(description='Shutter Image/Video Demo')
parser.add_argument('--partitions', type=int, default=1,
                    help='Number of partitions to split the dataset into, to run multiple jobs in parallel')
parser.add_argument('--part', type=int, default=0,
                    help='Partition number to download where 0 <= part < partitions')
parser.add_argument('--ngpus', type=int, default=1,
                    help='Number of gpus in a single node')
parser.add_argument('--gpu-id', type=int, default=0,
                    help='Number of gpus in a single node')
args = parser.parse_args()

shards = args.partitions * args.ngpus
shard_id = args.part * args.ngpus + args.gpu_id

model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_folder = '/mnt/data/data/roomroute/trajectory_shots_scamera_merge_log_p1/'
output_folder = '/mnt/data/data/roomroute/geoinformation_colmap_p1/'
all_vid_step3_json = glob.glob(f"{input_folder}/*step3.json")
all_vid_step3_json = sorted(all_vid_step3_json)

interval = len(all_vid_step3_json)//shards
starting = shard_id * interval
ending = (shard_id + 1)*interval if shard_id < shards - 1 else len(all_vid_step3_json)

# starting = 0
# ending = 10
paths = []
videoids = []
for file in all_vid_step3_json[starting:ending]:
    for gnode in json.load(open(file))['nodes']:
        model_name = gnode['model']
        video_shot = '-'.join(model_name.split('-')[:-1])
        video_id = '_'.join(video_shot.split('%')[0].split('_')[:-2])
        model_id = model_name.split('-')[-1]
        videoids.append(video_id)
        paths.append(f"/mnt/data/data/roomroute/trajectory_shots_scamera/{video_shot}/sparse/{model_id}/")

# pbar = tqdm(zip(paths, videoids))
# for path, sence_id in pbar:

def parse_file_to_dict(scene_id):
    result_dict = {}
    file_path = "/home/user/data/roomroute/blip_vqa_results_3fps_360p/" + scene_id + ".txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip():  # 忽略空行
                    key, value = line.strip().split(' ', 1)
                    result_dict[key] = value
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return result_dict

def filter_images(images,blip_labels):
    filtered_images = {}
    for image_id, image in images.items():
        if blip_labels[image["name"]] not in ["outside","backyard","driveway","porch","patio"]:
            filtered_images[image_id] = image
    return filtered_images

def get_parallax_ids(data_meta):
    path, sence_id = data_meta
    cur_parallax_ids = {}
    try:
        file_path = path   #TODO#
        #这种类型"/home/user/data/roomroute/trajectory_shots_scamera/h6fLXn1Am4g_180_280%270_370/sparse/0%0"
        # sence_id = path[52:63]   #TODO

        cameras, images, points3D = parse_images_bin(file_path)
        filter_points3D = filter_points(points3D)

        blip_labels = parse_file_to_dict(sence_id)
        images = filter_images(images,blip_labels)


        max_parallax_point3D_id = compute_max_parallax(filter_points3D, images)
        parallax_ids,idin_parallax_ids = compute_parallax_by_projection(max_parallax_point3D_id, filter_points3D, images)

        cur_parallax_ids = {'data_meta': data_meta, 'parallax_ids': [max_parallax_point3D_id, parallax_ids,idin_parallax_ids]}
    except Exception as e:
        print(e)
    return cur_parallax_ids




print("Starting MP get_parallax_ids")
with mp.Pool(10) as pool:
    res = pool.map(get_parallax_ids, zip(paths, videoids))

all_data_meta = [r['data_meta'] for r in res]
all_parallax_ids = [r['parallax_ids'] for r in res]

print("Starting SP get_depth_maps")
scale_factors = {}
# def get_scale_facor(data_meta, parallax_ids_meta):
for data_meta, parallax_ids_meta in tqdm(zip(all_data_meta, all_parallax_ids)):
    try:
        path, sence_id = data_meta
        max_parallax_point3D_id, parallax_ids,idin_parallax_ids = parallax_ids_meta

        file_path = path   #TODO#
        cameras, images, points3D = parse_images_bin(file_path)
        model_name = file_path.split('/')[-3] + '-' + file_path.split('/')[-1]
        image_path = os.path.join("/mnt/data/data/roomroute/trajectory_scamera",sence_id,"imgs_3fps_360p")
        

        idin_parallax_ids_with_depth = predict_depth_image(model, max_parallax_point3D_id, cameras, images, idin_parallax_ids)
        scale_factors_dict = compute_scale_factor(parallax_ids, idin_parallax_ids_with_depth, images, max_parallax_point3D_id, filter_points3D)
        # print(scale_factors_dict)
        scale_factors[model_name] = scale_factors_dict[0]
        dump_data = {'scales': scale_factors_dict, 'depths': idin_parallax_ids_with_depth}
        pkl.dump(dump_data, open(f'{output_folder}/video_shots_records/unit_scale_factor_depths_{model_name}.pkl', 'wb'))
        json.dump(scale_factors, open(f'{output_folder}/unit_scale_factor_{shard_id}_{shards}.json', 'w'))
    except Exception as e:
        # print(f"Dumping into {output_folder}/unit_scale_factor_{shard_id}_{shards}.json")
        # sparse_frame = get_sparse_frame(images, scale_factors_dict)

        # video_frames[file_path] = sparse_frame
        # with open('sparse_frame.json', 'w') as f:
        #     json.dump(video_frames, f, indent=4)
        # visualize_sparse_frame(sparse_frame,sence_id)
        print(e)
        continue
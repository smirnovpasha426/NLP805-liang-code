import pickle as pkl
import json
import copy
import argparse
import numpy as np
import multiprocessing as mp

# from common_utils import *
from read_write_models import read_model, write_model
from scipy.spatial.transform import Rotation as R
import numpy as np


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

def angle_between_cameras(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
    unit_v2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

# Above are Common utils
#######


#实现一个函数，依次计算每一个frame前后一系列帧和当前帧的距离，然后乘以尺度因子，将这些帧的序号、距离、以及和当前帧的夹角储存到一个新的字典中，字典的键是当前帧的frame_id，值是一个列表，列表中每一个元素是一个字典，字典中包含了前后帧的frame_id、距离和夹角
# def compute_frame_distance(images_data, scale_factors_dict, forward_distance=4.0, backward_distance=4.0):
def compute_frame_distance(input):
    model_name, forward_distance, backward_distance = input
    videoId = '-'.join(model_name.split('-')[:-1])
    modelId = model_name.split('-')[-1]
    file_path = f"/home/user/roomroute/data/roomroute_colmap/{videoId}/sparse/{modelId}"
    _, images_data, _ = parse_images_bin(file_path)
    scale_factors_dict = scale_factors[model_name]
    scale_factor = scale_factors_dict['mean']
    scale_factor_np = np.array(scale_factor)

    if scale_factors_dict['std'] >= 0.4*scale_factors_dict['mean']:
        return {}

    frames_data = {}
    for image_id in list(images_data.keys()):
        frame_id = images_data[image_id]["frame_id"]
        # frames_data["image_id"] = image_id
        if frame_id not in list(frames_data.keys()):
            frames_data[frame_id] = images_data[image_id]
            frames_data[frame_id]["image_id"] = image_id

    mem = {}
    frame_distance = {}
    frame_ids = list(frames_data.keys())
    frame_ids.sort()
    # last_frame_id = None
    for i in range(len(frame_ids)):
        frame_id = frame_ids[i]
        frame_distance[frame_id] = {}
        frame_distance[frame_id]["real_world_position"] = list(frames_data[frame_id]['camera_world_position'] * scale_factor_np)
        frame_distance[frame_id]["camera_world_position"] = frames_data[frame_id]['camera_world_position']
        frame_distance[frame_id]["quaternion"] = frames_data[frame_id]['quaternion']
        frame_distance[frame_id]["forward"] = {}
        frame_distance[frame_id]["backward"] = {}
        for j in range(i - 1, -1, -1):
            last_frame_id = frame_ids[j]
            # if f"{frame_id}|{last_frame_id}" in mem:
            #     frame_distance[frame_id]["backward"][last_frame_id] = mem[f"{frame_id}|{last_frame_id}"]
            #     continue
            # Calculate the translation distance between the current and last stored image
            translation_distance = frames_data[frame_id]['camera_world_position']
            last_translation_distance = frames_data[last_frame_id]['camera_world_position']
            distance = ((translation_distance[0] - last_translation_distance[0]) ** 2 +
                        (translation_distance[2] - last_translation_distance[2]) ** 2 ) ** 0.5
            # Apply the scale factor
            scale_factor = scale_factors_dict['mean']
            scaled_distance = distance * scale_factor
            if scaled_distance > backward_distance:
                break
            #计算两个图像的夹角
            angle = angle_between_cameras(frames_data[frame_id]["quaternion"], frames_data[last_frame_id]["quaternion"])
            last_frames_data = copy.deepcopy(frames_data[last_frame_id])
            # last_frames_data = {}
            last_frames_data["distance"] = scaled_distance
            last_frames_data["angle"] = angle
            scale_factor_np = np.array(scale_factor)
            last_frames_data["real_world_position"] = list(frames_data[last_frame_id]['camera_world_position'] * scale_factor_np)
            position_list = list(frames_data[last_frame_id]['camera_world_position'])
            last_frames_data["camera_world_position"] = position_list
            # last_frames_data["camera_world_position"] = list(frames_data[last_frames_data]['camera_world_position'])
            # frame_distance[frame_id].append(next_frames_data)
            if "xys" in last_frames_data:
                del last_frames_data["xys"]
            if "point3D_ids" in last_frames_data:
                del last_frames_data["point3D_ids"]
            # print(last_frames_data)
            frame_distance[frame_id]["backward"][last_frame_id] = last_frames_data
            # mem[f"{frame_id}|{last_frame_id}"] = last_frames_data


        for j in range(i + 1, len(frame_ids)):
            next_frame_id = frame_ids[j]
            # if f"{frame_id}|{next_frame_id}" in mem:
            #     frame_distance[frame_id]["forward"][next_frame_id] = mem[f"{frame_id}|{next_frame_id}"]
            #     continue
            # Calculate the translation distance between the current and last stored image
            translation_distance = frames_data[frame_id]['camera_world_position']
            next_translation_distance = frames_data[next_frame_id]['camera_world_position']
            distance = ((translation_distance[0] - next_translation_distance[0]) ** 2 +
                        (translation_distance[2] - next_translation_distance[2]) ** 2 ) ** 0.5
            # Apply the scale factor
            scale_factor = scale_factors_dict['mean']
            scaled_distance = distance * scale_factor
            if scaled_distance > forward_distance:
                break
            #计算两个图像的夹角
            angle = angle_between_cameras(frames_data[frame_id]["quaternion"], frames_data[next_frame_id]["quaternion"])
            next_frames_data = copy.deepcopy(frames_data[next_frame_id])
            # next_frames_data = {}
            next_frames_data['frame_id'] = next_frame_id
            next_frames_data["distance"] = scaled_distance
            next_frames_data["angle"] = angle
            scale_factor_np = np.array(scale_factor)
            next_frames_data["real_world_position"] = list(frames_data[next_frame_id]['camera_world_position'] * scale_factor_np)
            position_list = list(frames_data[next_frame_id]['camera_world_position'])
            next_frames_data["camera_world_position"] = position_list
            # next_frames_data["camera_world_position"] = list(frames_data[next_frame_id]['camera_world_position'])
            if "xys" in next_frames_data:
                del next_frames_data["xys"]
            if "point3D_ids" in next_frames_data:
                del next_frames_data["point3D_ids"]
            frame_distance[frame_id]["forward"][next_frame_id] = next_frames_data
            # mem[f"{frame_id}|{next_frame_id}"] = next_frames_data
    return {model_name: frame_distance}

parser = argparse.ArgumentParser(description='Shutter Image/Video Demo')
parser.add_argument('--partitions', type=int, default=1,
                    help='Number of partitions to split the dataset into, to run multiple jobs in parallel')
parser.add_argument('--part', type=int, default=0,
                    help='Partition number to download where 0 <= part < partitions')
# parser.add_argument('--ngpus', type=int, default=1,
#                     help='Number of gpus in a single node')
# parser.add_argument('--gpu-id', type=int, default=0,
#                     help='Number of gpus in a single node')
args = parser.parse_args()

shards = args.partitions # * args.ngpus
shard_id = args.part # * args.ngpus + args.gpu_id

all_frame_distance = {}
scale_factors = json.load(open('/home/user/roomroute/data/roomroute_geo/collated_32_shards.json'))
model_names = list(scale_factors.keys())
# videoIds = []
# modelIds = []
# for model_name in model_names:
#     videoId = '-'.join(model_name.split('-')[:-1])
#     modelId = model_name.split('-')[-1]
#     videoIds.append(videoId)
#     modelIds.append(modelId)

interval = len(model_names)//shards
starting = shard_id * interval
ending = (shard_id + 1)*interval if shard_id < shards - 1 else len(model_names)
print(starting, ending)
# compute_frame_distance(['DRTOnEwAkws_0_100%90_190-0%0', 2.5, 2.5])
# res = {}
# import random
# [res.update(compute_frame_distance(model_name)) for model_name in random.sample(model_names, 15)]

forward_radius = 2.5
backward_radius = 2.5
with mp.Pool(70) as pool:
    # for model_name in scale_factors:
    #计算所有帧前后
    # videoId = '-'.join(model_name.split('-')[:-1])
    # modelId = model_name.split('-')[-1]
    # file_path = f"/home/user/roomroute/data/roomroute_colmap/{videoId}/sparse/{modelId}"
    # _, images, _ = parse_images_bin(file_path)
    # all_frame_distance[model_name] = compute_frame_distance(images, scale_factors[model_name])
    inputs = [[model_name, forward_radius, backward_radius] for model_name in model_names[starting:ending]]
    res = pool.map(compute_frame_distance, inputs)

all_frame_distance = {}
_ = [all_frame_distance.update(r) for r in res]

pkl.dump(all_frame_distance, open(f'/home/user/roomroute/data/roomroute_geo/distance_and_degrees_{shard_id}_{shards}_2.5m.pkl', 'wb'))
# print(all_frame_distance)


def __init__(self, video_trajectory_dir):
    self.candidate_negative_frames_cnt = 6
    self.candidate_negative_interval = 20
    self.candidate_positive_frames_cnt = 2
    self.candidate_positive_interval = 2

    all_videos = glob.glob(video_trajectory_dir + '/*json')

    all_video_entries = {}
    all_distance_angles = {}
    last_videoId = None
    last_video_frames = None
    for i, file in enumerate(all_videos):
        vid = os.path.basename(file).replace('.json', '')
        vid_distance_angles = pkl.load(open("/home/user/roomroute/data/roomroute_distance_angle/{}.pkl", 'rb'))

        current_video_entries = {}
        for entry in json.load(open(file)):
            longId = entry["longId"]
            nextfid = entry["nextId"]
            colmapId = entry['scene']

            position = vid_distance_angles[colmapId][nextfid]
            current_video_entries[longId] = {'nextFrame': {'videoId': vid, 'frameId': f"output_frame_{nextfid:04d}.png"},
                                             'negativeFrame': [f"output_frame_{fid:04d}.png" for fid in entry['negativeIds']],
                                             'historyFrame': [f"output_frame_{fid:04d}.png" for fid in entry['history']],
                                             'futureFrame': [f"output_frame_{fid:04d}.png" for fid in entry['future']],
                                             'nav': [id==entry['nextId'] for id in entry['curIds']],
                                             'colmapId': entry['scene']
                                             }

        all_video_entries[vid] = current_video_entries

    self.cached_vid_sim = pkl.load(open('/home/user/roomroute/data/roomroute_connectivity/trajectory_paths.pkl', 'rb'))
    self.all_video_entries = all_video_entries

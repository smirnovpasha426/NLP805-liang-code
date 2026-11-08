import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(qw, qx, qy, qz):
    r = R.from_quat([qx, qy, qz, qw])
    return r.as_euler('xyz', degrees=True)  # Returns roll, pitch, yaw in degrees


def parse_images_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skipping the header lines and any comment lines
    lines = [line for line in lines if not line.startswith('#') and len(line.strip()) > 0]
    lines = [line for i, line in enumerate(lines) if i % 2 == 0]
    
    images_data = []
    for line in lines:
        parts = line.strip().split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        image_name = parts[9]
        roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz)
        
        images_data.append({
            'image_id': image_id,
            'quaternion': (qw, qx, qy, qz),
            'translation': (tx, ty, tz),
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'camera_id': camera_id,
            'image_name': image_name
        })
    
    return images_data

def analyze_camera_movements(camera_poses):
    movements = {'delta_yaw': [], 'delta_pitch': [], 'distance_moved': [], 'left_turns': [], 'right_turns': [], 'looking_up': [], 'looking_down': [], 'still': []}
    for i in range(1, len(camera_poses)):
        prev_pose = camera_poses[i-1]
        current_pose = camera_poses[i]

        # Calculate changes in orientation
        delta_yaw = current_pose['yaw'] - prev_pose['yaw']
        delta_pitch = current_pose['pitch'] - prev_pose['pitch']
        
        # Calculate linear movement
        distance_moved = np.linalg.norm(np.array(current_pose['translation']) - np.array(prev_pose['translation']))

        # Detecting turns and looking around
        if abs(delta_yaw) > 0.1 and distance_moved < 0.5:  # Adjust thresholds as needed
            if delta_yaw > 0:
                movements['right_turns'].append(i)
            else:
                movements['left_turns'].append(i)

        if abs(delta_pitch) > 0.1 and distance_moved < 0.5:  # Adjust thresholds as needed
            if delta_pitch > 0:
                movements['looking_down'].append(i)
            else:
                movements['looking_up'].append(i)

        if distance_moved < 0.02:  # Threshold for minimal movement, indicating viewing around a spot
            movements['still'].append(i)
        movements['delta_yaw'].append(delta_yaw)
        movements['delta_pitch'].append(delta_pitch)
        movements['distance_moved'].append(distance_moved)

    return movements

images_data = parse_images_txt('/Users/user/Documents/ytb_vln/demos/trajectory/8_living_room_into_kitchen/text/images.txt')
images_data = sorted(images_data, key=lambda x:x['image_id'])
analyze_camera_movements(images_data)
# movements = [analyze_camera_movements(d) for d in images_data]

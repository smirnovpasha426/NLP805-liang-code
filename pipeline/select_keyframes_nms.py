#縱軸——翻滾（roll），分逆時針翻（左翻）和順時針翻（右翻）； 橫軸——俯仰（pitch），分上仰和下俯； 垂軸——偏擺（yaw），分左擺和右擺
def compute_frame_distance(input):
    model_name, forward_distance, backward_distance = input
    videoId = '-'.join(model_name.split('-')[:-1])
    modelId = model_name.split('-')[-1]
    file_path = f"/mnt/data/data/roomroute/trajectory_shots_scamera/{videoId}/sparse/{modelId}"
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
            # angle = angle_between_cameras(frames_data[frame_id]["quaternion"], frames_data[last_frame_id]["quaternion"])
            angle = frames_data[frame_id]["yaw"] - frames_data[next_frame_id]["yaw"]
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
            # angle = angle_between_cameras(frames_data[frame_id]["quaternion"], frames_data[next_frame_id]["quaternion"])
            angle = frames_data[frame_id]["yaw"] - frames_data[next_frame_id]["yaw"]
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

def calculate_angle_diff(frame_data):
    if not frame_data:  # Check if frame_data is empty
        return 0
    Max = max(data["angle"] for _, data in frame_data.items())
    Min = min(data["angle"] for _, data in frame_data.items())
    return abs(Max - Min)
 
def get_window(frame_distance, frame_id):
    window = []
    if "forward" in frame_distance[frame_id]:
        for k, v in frame_distance[frame_id]["forward"].items():
            if v["distance"] < 2.5:
                window.append(k)
    if "backward" in frame_distance[frame_id]:
        for k, v in frame_distance[frame_id]["backward"].items():
            if v["distance"] < 2.5:
                window.append(k)
    return window
 
def non_maximum_suppression_1d_with_custom_window(angle_diffs, windows):
    suppressed = set()
    for i, (path, frame_id, angle_diff) in enumerate(angle_diffs):
        is_max = True
        current_window = windows[frame_id]
        for neighbor_id in current_window:
            neighbor_index = next((idx for idx, item in enumerate(angle_diffs) if item[1] == neighbor_id), None)
            if neighbor_index is not None and angle_diffs[neighbor_index][2] > angle_diff:
                is_max = False
                break
        if not is_max:
            suppressed.add((path, frame_id))
    return suppressed
 
Selected_all_frame_distance = {}
angle_diffs = []
windows = {}
 
# First pass: calculate angle differences and windows
for path, frame_distance in all_frame_distance.items():
    Selected_all_frame_distance[path] = {}
    for frame_id, frame_data in frame_distance.items():
        print(frame_data)
        angle_diff_f = calculate_angle_diff(frame_data["forward"])
        angle_diffs.append((path, frame_id, angle_diff_f))
        angle_diff_b = calculate_angle_diff(frame_data["backward"])
        angle_diffs.append((path, frame_id, angle_diff_b))
        windows[frame_id] = get_window(frame_distance, frame_id)
 
# Apply NMS
suppressed_frames = non_maximum_suppression_1d_with_custom_window(angle_diffs, windows)
 
# Remove suppressed frames from all_frame_distance
for path, frame_id in suppressed_frames:
    if frame_id in all_frame_distance[path]:
        del all_frame_distance[path][frame_id]
 
# Output the filtered all_frame_distance
print("删除前的数据", sum(len(fd) for fd in all_frame_distance.values()))
print("删除后的数据", sum(len(fd) for fd in all_frame_distance.values()))
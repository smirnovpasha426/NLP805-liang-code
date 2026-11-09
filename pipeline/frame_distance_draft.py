def compute_frame_distance(images_data, scale_factors_dict, forward_distance=4.0, backward_distance=4.0):
    frames_data = {}
    for image_id in list(images_data.keys()):
        frame_id = images_data[image_id]["frame_id"]
        # frames_data["image_id"] = image_id
        if frame_id not in list(frames_data.keys()):
            frames_data[frame_id] = images_data[image_id]
            frames_data[frame_id]["image_id"] = image_id

    frame_distance = {}
    frame_ids = list(frames_data.keys())
    frame_ids.sort()
    for i in range(len(frame_ids)):
        frame_id = frame_ids[i]
        frame_distance[frame_id] = {}
        frame_distance[frame_id]["forward"] = {}
        frame_distance[frame_id]["backward"] = {}
        for j in range(i - 1, -1, -1):
            last_frame_id = frame_ids[j]
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
            last_frames_data = frames_data[last_frame_id]
            last_frames_data["distance"] = scaled_distance
            last_frames_data["angle"] = angle
            scale_factor_np = np.array(scale_factor)
            last_frames_data["real_camera_world_position"] = frames_data[last_frame_id]['camera_world_position'] * scale_factor_np
            
            # last_frames_data["camera_world_position"] = list(frames_data[last_frames_data]['camera_world_position'])
            # frame_distance[frame_id].append(next_frames_data)
            if "xys" in last_frames_data:
                del last_frames_data["xys"]
            if "point3D_ids" in last_frames_data:
                del last_frames_data["point3D_ids"]
            # print(last_frames_data)
            frame_distance[frame_id]["backward"][last_frame_id] = last_frames_data

            
        for j in range(i + 1, len(frame_ids)):
            next_frame_id = frame_ids[j]
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
            next_frames_data = frames_data[next_frame_id]
            next_frames_data['frame_id'] = next_frame_id
            next_frames_data["distance"] = scaled_distance
            next_frames_data["angle"] = angle
            scale_factor_np = np.array(scale_factor)
            next_frames_data["real_camera_world_position"] = frames_data[next_frame_id]['camera_world_position'] * scale_factor_np
            # next_frames_data["camera_world_position"] = list(frames_data[next_frame_id]['camera_world_position'])
            if "xys" in next_frames_data:
                del next_frames_data["xys"]
            if "point3D_ids" in next_frames_data:
                del next_frames_data["point3D_ids"]
            frame_distance[frame_id]["forward"][last_frame_id] = next_frames_data
    return frame_distance

all_frame_distance = {}
for path in pbar:    
#计算所有帧前后
    _, images, _ = parse_images_bin(file_path)
    all_frame_distance[path] = compute_frame_distance(images, scale_factors[file_path])

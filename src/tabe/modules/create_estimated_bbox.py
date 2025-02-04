from typing import Optional

import cv2
import numpy as np

from src.tabe.utils.bbox_utils import calc_bbox_area
from src.tabe.utils.mask_utils import get_bbox_from_binary_mask
from src.tabe.utils.occlusion_utils import OcclusionLevel


# TODO NEED TO INVERT THE X AND Y AS THIS IS VERY CONFUSING

def get_corners_from_xyxy_bbox(bbox: np.ndarray | list) -> np.ndarray:
    y_min, x_min, y_max, x_max = bbox
    return np.array([
        [x_min, y_min],  # Top-left corner
        [x_max, y_min],  # Top-right corner
        [x_min, y_max],  # Bottom-left corner
        [x_max, y_max]  # Bottom-right corner
    ])


def check_for_occlusions_for_each_edge(bbox: np.ndarray, estimated_bbox: np.ndarray, in_front_obj: np.ndarray,
                                       behind_thresh: float = 0.99, side_thresh: int = 20) -> list:
    y, x, y2, x2 = bbox
    ey, ex, ey2, ex2 = estimated_bbox.copy()
    # Right of bbox
    area_behind_right = (in_front_obj[y: y2, x2: x2 + side_thresh] == 0).flatten()
    if (area_behind_right.sum() / len(area_behind_right)) > behind_thresh:
        ex2 = x2

    # top of bbox
    area_behind_top_close = (in_front_obj[y - side_thresh:y, x: x2] == 0).flatten()
    if (area_behind_top_close.sum() / len(area_behind_top_close)) > behind_thresh:
        ey = y

    # Left
    area_behind_left_close = (in_front_obj[y: y2, x - side_thresh: x] == 0).flatten()
    if (area_behind_left_close.sum() / len(area_behind_left_close)) > behind_thresh:
        ex = x

    # Bottom
    area_behind_bottom_close = (in_front_obj[y2: y2 + side_thresh, x: x2] == 0).flatten()
    if (area_behind_bottom_close.sum() / len(area_behind_bottom_close)) > behind_thresh:
        ey2 = y2

    return [ey, ex, ey2, ex2]


def find_min_point_change_in_bboxes(bbox_1: np.ndarray, bbox_2: np.ndarray, points_occluded: bool = None) -> tuple[
    float, int]:
    distances = np.linalg.norm(bbox_1 - bbox_2, axis=1)

    if points_occluded is not None and not np.all(points_occluded):
        # If not all points are counted as occluded we can remove these ones from contention
        distances[points_occluded] = 1e9

    # Find the indices of the minimum distance
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    return min_distance, min_idx


def corners_to_xyxy(corners: np.ndarray) -> list:
    x_min = np.min(corners[:, 1])
    y_min = np.min(corners[:, 0])
    x_max = np.max(corners[:, 1])
    y_max = np.max(corners[:, 0])
    return [x_min, y_min, x_max, y_max]


def include_all_of_other_bbox(bbox: list, other_bbox: list) -> list:
    new_x1 = int(min(bbox[0], other_bbox[0]))
    new_y1 = int(min(bbox[1], other_bbox[1]))
    new_x2 = int(max(bbox[2], other_bbox[2]))
    new_y2 = int(max(bbox[3], other_bbox[3]))

    return [new_x1, new_y1, new_x2, new_y2]


def interpolate_bounding_boxes(box1: list, box2: list, n_frames: int) -> list:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate step sizes for each corner
    step_x_min = (x2_min - x1_min) / (n_frames + 1)
    step_y_min = (y2_min - y1_min) / (n_frames + 1)
    step_x_max = (x2_max - x1_max) / (n_frames + 1)
    step_y_max = (y2_max - y1_max) / (n_frames + 1)

    # Generate interpolated bounding boxes
    interpolated_boxes = []
    for i in range(1, n_frames + 1):
        xi_min = x1_min + i * step_x_min
        yi_min = y1_min + i * step_y_min
        xi_max = x1_max + i * step_x_max
        yi_max = y1_max + i * step_y_max
        interpolated_boxes.append((xi_min, yi_min, xi_max, yi_max))

    return interpolated_boxes


def split_into_chunks(arr: np.ndarray) -> list:
    zero_indices = np.where(arr == 0)[0]  # Indices where element is zero
    zero_chunks = []

    # Group consecutive indices into chunks
    chunk = [zero_indices[0]]
    for i in range(1, len(zero_indices)):
        if zero_indices[i] == zero_indices[i - 1] + 1:
            chunk.append(zero_indices[i])
        else:
            zero_chunks.append(chunk)
            chunk = [zero_indices[i]]
    zero_chunks.append(chunk)  # Add the last chunk

    return zero_chunks


def does_curr_bbox_provide_enough_info(bbox: Optional[list], min_bbox_area: int = 5):
    if bbox is None:
        return False
    return calc_bbox_area(*bbox) > min_bbox_area


def get_motion_estimate(estimated_bbox_xyxy: list, frame_num: int, last_x_frames: int = 2) -> np.ndarray:
    # We know it is in the frame but have no box at all, so just have to use previous motion
    last_x_frames_motion = []
    for fn in range(frame_num - 1, frame_num - last_x_frames, -1):
        if fn not in estimated_bbox_xyxy or (fn - 1 not in estimated_bbox_xyxy):
            break
        prev_bbox = estimated_bbox_xyxy[fn - 1].copy()
        if prev_bbox is None:
            continue
        prev_y, prev_x, prev_y2, prev_x2 = prev_bbox

        curr_bbox = estimated_bbox_xyxy[fn].copy()
        if curr_bbox is None:
            continue
        y, x, y2, x2 = curr_bbox
        y_diffs = [y - prev_y, y2 - prev_y2]
        min_y_dist_index = np.argmin(np.abs(y_diffs))
        x_diffs = [x - prev_x, x2 - prev_x2]
        min_x_dist_index = np.argmin(np.abs(x_diffs))
        last_x_frames_motion.append(np.array([y_diffs[min_y_dist_index], x_diffs[min_x_dist_index]]))
    if not len(last_x_frames_motion):
        motion_estimate = np.array([0, 0])
    else:
        motion_estimate = np.array(last_x_frames_motion).mean(0)
    return motion_estimate


def estimate_bboxes(masks: np.ndarray, occlusion_info: list[OcclusionLevel], monodepth_results: list[dict],
                    allow_look_ahead: bool = True) -> dict:
    masks = masks.copy()  # Dont want to change for other parts of the pipeline
    for i, mask in enumerate(masks):
        if mask.max() > 1:
            masks[i] = mask // 255

    bbox_info = {}
    for i, mask in enumerate(masks):
        occl_info = occlusion_info[i]
        if mask is None:
            bbox_info[i] = [None, occl_info]
            continue
        bbox = get_bbox_from_binary_mask(mask)
        bbox_info[i] = [bbox, occl_info]

    estimated_bbox_xyxy = {}
    prev_estimated_bbox = None
    for frame_num in bbox_info:
        if bbox_info[frame_num][1] == OcclusionLevel.NO_OCCLUSION:
            prev_estimated_bbox = bbox_info[frame_num][0]
            estimated_bbox_xyxy[frame_num] = prev_estimated_bbox
            continue
        vis_bbox = bbox_info[frame_num][0]
        depth_img = monodepth_results[frame_num]["depth"]
        depth_img_arr = np.array(depth_img)
        depth_vals_for_object = depth_img_arr[masks[frame_num] == 1]
        avg_depth_val = depth_vals_for_object.mean()
        in_front_obj = np.zeros_like(depth_img_arr)
        in_front_obj[depth_img_arr > avg_depth_val] = 255
        in_front_obj[masks[frame_num] == 1] = 127
        if bbox_info[frame_num][0] is not None and does_curr_bbox_provide_enough_info(vis_bbox):
            # If we have any sort of mask, even if its leaving scene it can help us map the motion of the object
            bbox_corners_prev = get_corners_from_xyxy_bbox(prev_estimated_bbox)
            bbox_corners_curr = get_corners_from_xyxy_bbox(vis_bbox)
            points_likely_occluded = []
            # Find which of the current corner points are likely occluded
            for corner in bbox_corners_curr:
                expanded_points = np.zeros_like(in_front_obj)
                cv2.circle(expanded_points, corner, 3, 1, -1)
                points_likely_occluded.append((in_front_obj[expanded_points == 1] == 255).any())

            min_dist, closest_corner = find_min_point_change_in_bboxes(bbox_corners_prev, bbox_corners_curr,
                                                                       points_likely_occluded)
            closest_corner_prev, closest_corner_curr = closest_corner, closest_corner

            xy_dist = bbox_corners_prev[closest_corner_prev] - bbox_corners_curr[closest_corner_curr]
            est_bbox_with_corners = bbox_corners_prev - xy_dist
            estimated_bbox_xyxy[frame_num] = corners_to_xyxy(est_bbox_with_corners)
            # And it must include all ground truth
            estimated_bbox_xyxy[frame_num] = include_all_of_other_bbox(estimated_bbox_xyxy[frame_num], vis_bbox)

            # If edge has grown but no occlusion we want to remove that edge being occluded and set to vis box
            occl_removed_est_bbox = check_for_occlusions_for_each_edge(vis_bbox, estimated_bbox_xyxy[frame_num],
                                                                       in_front_obj, behind_thresh=30)
            estimated_bbox_xyxy[frame_num] = occl_removed_est_bbox
        else:
            motion_estimate = get_motion_estimate(estimated_bbox_xyxy, frame_num)

            # if we have the previous ground truth non occluded use this, else use last predicted mask
            estimated_bbox_xyxy[frame_num] = prev_estimated_bbox.copy()
            estimated_bbox_xyxy[frame_num][0] += motion_estimate[0]
            estimated_bbox_xyxy[frame_num][1] += motion_estimate[1]
            estimated_bbox_xyxy[frame_num][2] += motion_estimate[0]
            estimated_bbox_xyxy[frame_num][3] += motion_estimate[1]
            estimated_bbox_xyxy[frame_num] = np.array(estimated_bbox_xyxy[frame_num]).astype(int)

        prev_estimated_bbox = estimated_bbox_xyxy[frame_num].copy()

    # Now finally we can just interpolate the missing frames
    if allow_look_ahead:
        mask_sums = np.array(masks).reshape(len(masks), -1).sum(-1)
        too_small_mask = mask_sums == 0
        if too_small_mask.any():
            chunks = split_into_chunks(~too_small_mask)
            for zero_chunk in chunks:
                # +-1 as we want the prev and after non zero bbox, although if final frame is fully occluded we have to
                # just take the largest
                first_frame = max(0, zero_chunk[0] - 1)
                last_frame = zero_chunk[-1] + 1
                if last_frame > max(estimated_bbox_xyxy):
                    last_frame = max(estimated_bbox_xyxy)
                interpolated_bboxes = interpolate_bounding_boxes(estimated_bbox_xyxy[first_frame],
                                                                 estimated_bbox_xyxy[last_frame], len(zero_chunk))
                for i, fn in enumerate(range(zero_chunk[0], zero_chunk[-1] + 1)):
                    estimated_bbox_xyxy[fn] = np.array(interpolated_bboxes[i]).astype(int)

        # Also interpolate each frame from previous and next
        for frame_num in estimated_bbox_xyxy:
            if frame_num == min(estimated_bbox_xyxy.keys()) or frame_num == max(estimated_bbox_xyxy.keys()):
                continue
            interpolated_bbox = interpolate_bounding_boxes(estimated_bbox_xyxy[frame_num - 1],
                                                           estimated_bbox_xyxy[frame_num + 1], 1)[0]
            estimated_bbox_xyxy[frame_num] = include_all_of_other_bbox(estimated_bbox_xyxy[frame_num],
                                                                       interpolated_bbox)

    return estimated_bbox_xyxy

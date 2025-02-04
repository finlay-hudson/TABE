from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import LineString, Polygon
from skimage.draw import line
from scipy.ndimage import binary_erosion, binary_dilation, median_filter

from src.tabe.utils.mask_utils import get_bbox_from_binary_mask


class OcclusionLevel(Enum):
    SEVERE_OCCLUSION = 0
    SLIGHT_OCCLUSION = 1
    NO_OCCLUSION = 2
    NONE = 3


@dataclass
class OcclusionInfo:
    level: OcclusionLevel
    amount: Optional[float] = None


def map_occlusion_levels(occl_infos: list[str | int], amount_of_occlusion: Optional[list[float]] = None) -> list[
    OcclusionInfo]:
    occl_levels = []
    for i, occL_lvl in enumerate(occl_infos):
        amount_of_occl = amount_of_occlusion[i] if amount_of_occlusion is not None else None
        if occL_lvl == 3 or occL_lvl is None or occL_lvl == "None" or occL_lvl == "NONE":
            mapped_lvl = OcclusionLevel.NONE
        else:
            occL_lvl = str(occL_lvl)
            if occL_lvl == "0" or occL_lvl.lower().startswith("severe"):
                mapped_lvl = OcclusionLevel.SEVERE_OCCLUSION
            elif occL_lvl == "1" or occL_lvl.lower().startswith("slight"):
                mapped_lvl = OcclusionLevel.SLIGHT_OCCLUSION
            elif occL_lvl == "2" or occL_lvl.lower().startswith("no"):
                mapped_lvl = OcclusionLevel.NO_OCCLUSION
            else:
                raise ValueError(f"Unknown occlusion info: {occL_lvl}")
        occl_levels.append(OcclusionInfo(level=mapped_lvl, amount=amount_of_occl))

    return occl_levels


def find_occlusion_level(masks: list, is_occluded: list, chunk_size: int = 10, severe_coef: int = 3):
    if isinstance(is_occluded, dict):
        is_occluded = list(is_occluded.values())
    is_occluded = np.array(is_occluded)
    mask_sums = np.stack(masks).reshape(len(masks), -1).sum(-1)
    # colors = ['green' if occl else 'red' for occl in is_occluded]
    # plt.scatter(range(len(mask_sums)), mask_sums, color=colors)

    occlusion_levels = np.ones(len(mask_sums), dtype=np.uint8) * 2  # default to no occlusion
    occlusion_levels[is_occluded] = 1  # set to slight occlusion
    prev_non_occluded_mean = 0
    for i in range(0, len(mask_sums), chunk_size):
        chunk_sum = mask_sums[i:i + chunk_size]
        chunk_occl = is_occluded[i:i + chunk_size]
        if not (~chunk_occl).any():
            # raise ValueError("Figure what to do here as maybe just have to take previous max size or something!!!")
            non_occluded_mean = prev_non_occluded_mean
        else:
            non_occluded_mean = chunk_sum[~chunk_occl].mean()
        severely_occluded = chunk_sum < (non_occluded_mean / severe_coef)
        # severely_occluded[chunk_sum == 0] = False  # We want to leave this as None and not just severely occluded
        occlusion_levels[i:i + chunk_size][severely_occluded] = 0
        occlusion_levels[i:i + chunk_size][chunk_sum == 0] = 0

    return occlusion_levels


def contour_area(contour):
    """ Calculate the area of a contour using its convex hull. """
    if len(contour) < 3:
        return 0.0
    polygon = Polygon(contour)
    return polygon.area


def filter_small_contours(contours: list, min_area: int = 5) -> list:
    """ Filter out contours with an area smaller than min_area. """
    return [contour for contour in contours if contour_area(contour) > min_area]


def calculate_distance_threshold(distances: list, percentile: int = 70) -> np.ndarray:
    """ Dynamically set the distance threshold based on the top percentile of distances. """
    distances_sorted = np.sort(distances)
    index = int(len(distances_sorted) * (percentile / 100.0)) - 1

    return distances_sorted[index]


def connect_contours(mask: np.ndarray, contours: list, max_dist: int = 500) -> np.ndarray:
    result_image = np.zeros_like(mask, dtype=np.uint8)
    # Create KD-Trees for each contour
    trees = [cKDTree(contour) for contour in contours]

    all_distances = []
    all_connections = []
    for i, (contour1, tree1) in enumerate(zip(contours, trees)):
        for j, (contour2, tree2) in enumerate(zip(contours, trees)):
            if i != j:
                # For contour2, find nearest points in contour1
                dists, nearest_indices = tree2.query(contour1)

                # Filter by max_dist
                close_mask = dists < max_dist
                close_points = contour1[close_mask]
                close_nearest_points = contour2[nearest_indices[close_mask]]
                close_dists = dists[close_mask]

                if not len(close_points) or not len(close_nearest_points):
                    continue

                lines_to_check = []
                for pt1, pt2 in zip(close_points, close_nearest_points):
                    lines_to_check.append(LineString([pt1, pt2]))
                lines_to_check = np.stack(lines_to_check)

                contour_lines = [LineString([contour1[i], contour1[i + 1]]) for i in range(len(contour1) - 1)]
                contour_lines.append(LineString([contour1[-1], contour1[0]]))

                intersects = shapely.intersects(lines_to_check[:, None], contour_lines)
                touches = shapely.touches(lines_to_check[:, None], contour_lines)
                valid_connection = ~(intersects & ~touches).any(-1)
                valid_distances = close_dists[valid_connection]
                valid_close_points = close_points[valid_connection]
                valid_close_nearest_points = close_nearest_points[valid_connection]
                valid_all_connections = [(tuple(pt1), tuple(pt2)) for pt1, pt2 in zip(valid_close_points,
                                                                                      valid_close_nearest_points)]

                all_distances.extend(valid_distances)
                all_connections.extend(valid_all_connections)

    if len(all_distances):
        # Set dynamic distance threshold based on top 70% of distances
        # distance_threshold = calculate_distance_threshold(all_distances, percentile=70)
        distance_threshold = calculate_distance_threshold(all_distances, percentile=99)

        for i, (point, closest_point) in enumerate(all_connections):
            if all_distances[i] < distance_threshold:
                create_line_img(result_image, point, closest_point)

    return result_image


def create_line_img(img: np.ndarray, pt1: list, pt2: list) -> None:
    rr, cc = line(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))
    img[rr, cc] = 1


def find_closest_from_different_masks(mask1: np.ndarray, mask2: np.ndarray) -> dict:
    # Get coordinates of non-zero pixels in mask2
    non_zero_coords_mask2 = np.argwhere(mask2 != 0)

    # Get coordinates of zero pixels in mask1
    non_zero_coords_mask1 = np.argwhere(mask1 != 0)

    # Build a KD-Tree for the non-zero coordinates from mask2
    tree = cKDTree(non_zero_coords_mask2)

    # Find the 5 closest non-zero neighbors from mask2 for all zero pixels in mask1
    distances, indices = tree.query(non_zero_coords_mask1, k=5)

    # Lookup values of the nearest non-zero pixels in mask2
    closest_values = {}
    for i, zero_coord in enumerate(non_zero_coords_mask1):
        # Get the coordinates of the 5 closest non-zero points
        closest_coords = non_zero_coords_mask2[indices[i][:len(non_zero_coords_mask2)]]
        # Get the values at those closest coordinates
        closest_vals = [mask2[tuple(coord)] for coord in closest_coords]
        # Map the zero-valued pixel to the 5 closest values
        closest_values[tuple(zero_coord)] = closest_vals

    return closest_values


def find_nearest_pixels(shrunk_mask: np.ndarray, expanded_mask: np.ndarray, min_dist: int = 3) -> np.ndarray:
    # Find the coordinates of non-zero pixels (True pixels) in the expanded mask
    expanded_coords = np.argwhere(expanded_mask)

    # Initialize an array to store the nearest values in the expanded mask
    in_front_vals = np.zeros_like(shrunk_mask, dtype=np.float32)

    for (i, j) in np.argwhere(shrunk_mask):
        # Calculate distances from the current pixel to all non-zero pixels in the expanded mask
        distances = np.sqrt((expanded_coords[:, 0] - i) ** 2 + (expanded_coords[:, 1] - j) ** 2)

        # Get the index of the closest expanded pixel
        nearest_index = np.argmin(distances)
        in_front_vals[i, j] = expanded_mask[
                                  expanded_coords[nearest_index][0], expanded_coords[nearest_index][1]] > (
                                      shrunk_mask[i, j] + min_dist)

    return in_front_vals


def assert_if_something_in_front(img_mask: np.ndarray, frame_monodepth_results: dict,
                                 ref_mask: Optional[np.ndarray] = None, thresh: float = 0.01, radius: int = 5,
                                 min_dist: int = 3, use_bbox: bool = False, prev_occl: bool = False) -> tuple[
    bool, float]:
    depth_img_arr = np.array(frame_monodepth_results["depth"])
    depth_img_arr[img_mask == 1] = 0
    if use_bbox:
        mask_bbox = get_bbox_from_binary_mask(img_mask)
        depth_img_arr = depth_img_arr[mask_bbox[0]:mask_bbox[2], mask_bbox[1]:mask_bbox[3]]
    else:
        exp_mask = cv2.dilate(img_mask.astype(np.uint8).copy(), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        contours, _ = cv2.findContours(exp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Put them in skimage format
        contours = [contour[:, 0, ::-1] for contour in contours]
        contours = filter_small_contours(contours, min_area=250)
        if len(contours) > 1:
            result_image = connect_contours(img_mask, contours)
            exp_mask = (exp_mask | result_image)
        # exp_mask = cv2.dilate(exp_mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        depth_img_arr[exp_mask == 0] = 0

    if ref_mask is not None:
        depth_img_arr[ref_mask == 0] = 0

    depth_map_of_mask = np.array(frame_monodepth_results["depth"])
    # Clean up any outliers in the depth map
    depth_map_of_mask = median_filter(depth_map_of_mask, size=3)

    shrunk_mask = binary_erosion(img_mask, structure=np.ones((2, 2), dtype=np.uint8), iterations=1)
    shrunk_outline = cv2.Canny((shrunk_mask * 255).astype(np.uint8), 100, 200)
    shunk_depth_map_of_mask = depth_map_of_mask.copy()
    shunk_depth_map_of_mask[shrunk_outline == 0] = 0

    expanded_mask = binary_dilation(img_mask, structure=np.ones((3, 3), dtype=np.uint8), iterations=1)
    expanded_outline = cv2.Canny((expanded_mask * 255).astype(np.uint8), 100, 200)
    expanded_depth_map_of_mask = depth_map_of_mask.copy()
    expanded_depth_map_of_mask[expanded_outline == 0] = 0

    if ((shunk_depth_map_of_mask > 0).sum() < 20) and prev_occl:
        # The mask must be so small that it is not possible to find a nearest pixel so we assume occluded
        return True, 1e9

    result = find_nearest_pixels(shunk_depth_map_of_mask, expanded_depth_map_of_mask, min_dist=min_dist)
    occl_val = np.sum(result) / (shunk_depth_map_of_mask > 0).sum()

    return occl_val > thresh, occl_val


def find_if_any_monodepth_occlusion(manual_masks: np.ndarray | list, monodepth_results: list,
                                    md_occl_thresh: float = 0.01) -> tuple[dict, dict]:
    is_item_occluded = {}
    amounts_of_occlusion = {}
    prev_occl_pred = False
    for annotated_frame_id in range(0, len(manual_masks)):
        manual_mask = manual_masks[annotated_frame_id]
        manual_mask = manual_mask.clip(0, 1)
        occlusion_pred = True
        if manual_mask.sum() == 0:
            amount_of_occlusion = 1e9
        else:
            occlusion_pred, amount_of_occlusion = assert_if_something_in_front(manual_mask,
                                                                               monodepth_results[annotated_frame_id],
                                                                               thresh=md_occl_thresh, radius=5,
                                                                               prev_occl=prev_occl_pred)
            prev_occl_pred = occlusion_pred
        is_item_occluded[annotated_frame_id] = occlusion_pred
        amounts_of_occlusion[annotated_frame_id] = amount_of_occlusion

    return is_item_occluded, amounts_of_occlusion

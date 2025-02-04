import numpy as np

from src.tabe.modules.components.item_leaving_frame import is_item_leaving_frame
from src.tabe.utils.occlusion_utils import find_if_any_monodepth_occlusion, find_occlusion_level


def predict_occlusion(frames: np.ndarray, masks: list | np.ndarray, monodepth_results, query_frame: bool = None,
                      md_occl_thresh: float = 0.01) -> tuple[list[int], list[float]]:
    if isinstance(masks, list):
        _masks = []
        for m in masks:
            _masks.append(m if m is not None else np.zeros(frames.shape[1:3], np.uint8))
        masks = np.stack(_masks)
    is_mask_empty = masks.reshape(masks.shape[0], -1).sum(-1) == 0
    is_item_occluded, amount_of_occlusion = find_if_any_monodepth_occlusion(masks, monodepth_results,
                                                                            md_occl_thresh=md_occl_thresh)
    item_heading_out_of_frame = True
    item_leaving_per_frame = {}
    for annotated_frame_id in range(0, len(frames)):
        item_leaving_per_frame[annotated_frame_id] = is_item_leaving_frame(masks, annotated_frame_id,
                                                                           prev_out_of_frame=item_heading_out_of_frame,
                                                                           edge_of_scene_thresh=1)
        item_heading_out_of_frame = item_leaving_per_frame[annotated_frame_id]

    visible_frame = ~np.array(list(is_item_occluded.values())) & ~np.array(
        list(item_leaving_per_frame.values())) & ~is_mask_empty
    if query_frame is not None:
        if not visible_frame.any():
            raise ValueError("Cannot determine a query frame")
        query_frame = np.where(visible_frame)[0][0]

    occlusion_levels = find_occlusion_level(masks[query_frame:], list(is_item_occluded.values())[query_frame:])
    occlusion_levels[list(item_leaving_per_frame.values())] = 3  # We assume out of frame so these should be set to None

    return occlusion_levels, amount_of_occlusion

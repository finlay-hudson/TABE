from src.tabe.utils.mask_utils import get_bbox_from_binary_mask


def is_item_leaving_frame(all_masks, frame_idx: int, prev_out_of_frame: bool = False,
                          edge_of_scene_thresh: int = 10) -> bool:
    if all_masks[frame_idx] is None or all_masks[frame_idx].sum() == 0:
        # We just have to trust the previous out of frame state
        return prev_out_of_frame
    mask_shape = all_masks[frame_idx].shape
    curr_bbox = get_bbox_from_binary_mask(all_masks[frame_idx])
    if (curr_bbox[0] <= 0 or curr_bbox[1] <= 0 or curr_bbox[2] >= (mask_shape[0] - edge_of_scene_thresh) or
            curr_bbox[3] >= (mask_shape[1] - edge_of_scene_thresh)):
        if prev_out_of_frame:
            return True
        if (frame_idx + 1) < len(all_masks):
            next_mask = all_masks[frame_idx + 1]
            if next_mask is None or next_mask.sum() == 0:
                # Have to assume it has gone out of the frame
                return True
            next_bbox = get_bbox_from_binary_mask(all_masks[frame_idx + 1])
            if (next_bbox[0] <= 0 or next_bbox[1] <= 0 or
                    next_bbox[2] >= (mask_shape[0] - edge_of_scene_thresh) or
                    next_bbox[3] >= (mask_shape[1] - edge_of_scene_thresh)):
                return True
            # Might have just touched the edge of the frame for one frame so just assume not

    return False

import cv2
import numpy as np
from pycocotools import mask as coco_mask
import torch


def rle_to_bitmap(rle):
    if rle is None:
        return None
    return coco_mask.decode(rle)


def mask_to_rle(mask):
    if mask is None:
        return None
    return coco_mask.encode(mask)


def compute_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """
    Compute Intersection over Union (IoU) for two binary masks.

    Args:
    - mask1 (torch.Tensor): First binary mask.
    - mask2 (torch.Tensor): Second binary mask.

    Returns:
    - float: IoU score.
    """
    intersection = torch.sum((mask1 & mask2).float())
    union = torch.sum((mask1 | mask2).float())
    iou = intersection / union
    return iou.item()


def get_bbox_from_binary_mask(binary_mask):
    # Find the non-zero coordinates
    non_zero_coords = np.argwhere(binary_mask)

    # If there are no non-zero pixels, return None
    if non_zero_coords.size == 0:
        return None

    # Get the bounding box coordinates
    min_row, min_col = np.min(non_zero_coords, axis=0)
    max_row, max_col = np.max(non_zero_coords, axis=0)

    # Return the bounding box as (min_row, min_col, max_row, max_col)
    return np.array([min_row, min_col, max_row, max_col]).astype(int)


def scale_up_mask(mask, iters=3, kernel_size=5):
    return cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=iters)


def move_pixels_by_xy(mask, num_right, num_down):
    num_down = int(num_down)
    num_right = int(num_right)
    shifted_mask = np.roll(mask, shift=(num_down, num_right), axis=(0, 1))

    # Zero out the wrapped-around areas
    if num_down > 0:
        shifted_mask[:num_down, :] = 0
    elif num_down < 0:
        shifted_mask[num_down:, :] = 0

    if num_right > 0:
        shifted_mask[:, :num_right] = 0
    elif num_right < 0:
        shifted_mask[:, num_right:] = 0

    return shifted_mask


def convert_one_to_three_channel(img):
    if img.ndim > 2:
        return img
    return np.stack((img,) * 3, axis=-1)


def convert_masks_to_correct_format(masks):
    masks = masks.astype(np.uint8)
    if masks.ndim == 4:
        masks = masks[..., 0]  # Convert to a single channel
    if masks.max() <= 1:
        masks *= 255

    return masks

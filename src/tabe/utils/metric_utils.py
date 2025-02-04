import math
import numpy as np


def batch_iou(masks1: np.ndarray, masks2: np.ndarray) -> np.ndarray:
    """
    Calculate the IoU between two batches of binary masks.

    Parameters:
    - masks1: np.ndarray of shape (N, H, W), where N is the batch size, H and W are the height and width of each mask.
    - masks2: np.ndarray of shape (N, H, W), where N is the batch size, H and W are the height and width of each mask.

    Returns:
    - iou_scores: np.ndarray of shape (N,), IoU scores for each pair of masks in the batch.
    """
    # Ensure that both masks have the same shape
    assert masks1.shape == masks2.shape, "Mask batches must have the same shape"

    # Allow for some elements to be NaN if they want to be ignored in the iou output
    nan_mask = np.isnan(np.array(masks1)).any(axis=(-1, -2)) | np.isnan(masks2).any(axis=(-1, -2))

    # Calculate the intersection and union
    intersection = np.logical_and(masks1, masks2).sum(axis=(-1, -2))
    union = np.logical_or(masks1, masks2).sum(axis=(-1, -2))

    # Avoid division by zero and calculate IoU
    iou_scores = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

    iou_scores[nan_mask] = np.nan

    return iou_scores


def get_non_vis_iou(pred_masks: np.ndarray, gt_amodal_masks: np.ndarray, gt_vis_masks: np.ndarray) -> np.ndarray:
    # Find IoU for occluded pixels by removing visible pixels from masks
    if pred_masks.max() > 1:
        pred_masks = pred_masks // 255
    if gt_amodal_masks.max() > 1:
        gt_amodal_masks = gt_amodal_masks.copy() // 255
    if gt_vis_masks.max() > 1:
        gt_vis_masks = gt_vis_masks.copy() // 255
    non_vis_mask_gt = gt_amodal_masks - gt_vis_masks
    non_vis_mask_pred = pred_masks - gt_vis_masks

    return batch_iou(non_vis_mask_gt, non_vis_mask_pred)


def get_metrics_per_method(data: dict) -> dict:
    method_metric_values = {}

    # Iterate through each video
    for video, methods in data.items():
        # Iterate over each method in the video
        for method, metrics in methods.items():
            # Initialize a dictionary for each method if not already present
            if method not in method_metric_values:
                method_metric_values[method] = {}

            # Collect values for each metric in the current method
            for metric, value in metrics.items():
                if metric not in method_metric_values[method]:
                    method_metric_values[method][metric] = []
                # Add the value if it's not NaN
                if not (isinstance(value, float) and math.isnan(value)):
                    method_metric_values[method][metric].append(value)

    # Calculate the mean for each method and metric across all videos
    return {
        method: {metric: np.nanmean(values) if values else None for metric, values in metrics.items()}
        for method, metrics in method_metric_values.items()
    }

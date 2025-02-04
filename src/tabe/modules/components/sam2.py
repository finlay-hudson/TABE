# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Altered to allow for frames to just be inputted and not needing a directory
"""

from collections import OrderedDict
import types
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


@torch.inference_mode()
def init_state(
        self,
        images,
        video_height,
        video_width,
        compute_device,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
):
    """Initialize an inference state."""
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # A storage to hold the model's tracking results and states on each frame
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
    }
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),  # set containing frame indices
        "non_cond_frame_outputs": set(),  # set containing frame indices
    }
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}

    # Warm up the visual backbone and cache the image feature on frame 0
    self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def sample_points(mask_to_use: np.ndarray, max_sample_points: int):
    pos_points = np.column_stack(np.where(mask_to_use)[::-1])
    if (max_sample_points > 0) and (len(pos_points) > max_sample_points):
        sampled_indices = sorted(np.random.choice(pos_points.shape[0], size=max_sample_points, replace=False))
        pos_points = pos_points[sampled_indices]
    pos_labels = np.ones(len(pos_points))

    return pos_points, pos_labels


def get_sam2_predictor(single_image: bool = False, model_cfg: str = "sam2_hiera_l.yaml",
                       checkpoint: str | Path = "sam2_hiera_large.pt", device: torch.device = torch.device("cuda")):
    if single_image:
        return SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    pred = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    # Monkey patch the new init state method to allow for frames to just be fed to the method
    pred.init_state = types.MethodType(init_state, pred)

    return pred


def predict_masks_sam(sam_pred, query_mask, running_frames, img_shape):
    inference_state = sam_pred.init_state(running_frames, img_shape[0], img_shape[1], sam_pred.device)
    _, _, out_mask_logits = sam_pred.add_new_mask(inference_state, 0, 1, query_mask)
    video_segments = [(out_mask_logits[0, 0] > 0.0).cpu().numpy() for _, _, out_mask_logits in
                      sam_pred.propagate_in_video(inference_state)]
    sam_pred.reset_state(inference_state)

    return video_segments


def sam_preprocessing(frames: List[Image.Image], mask: Image.Image, resize_to: Optional[tuple] = None) -> Tuple[
    torch.Tensor, torch.Tensor]:
    if not len(frames):
        raise ValueError("No frames given")
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    frame_tens_list = [pil_to_tensor(f.resize(resize_to) if resize_to is not None else f) for f in frames]

    frames_tens = torch.stack(frame_tens_list)
    frames_tens = frames_tens / 255.0

    # normalize by mean
    frames_tens -= torch.tensor(img_mean, dtype=torch.float32)[:, None, None]

    # normalize by std
    frames_tens /= torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    return frames_tens, torch.from_numpy(np.array(mask.resize(resize_to) if resize_to is not None else mask))

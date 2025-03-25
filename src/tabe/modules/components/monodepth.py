from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from transformers import pipeline as tr_pipeline


class MonoDepth:
    def __init__(self, checkpoint: str | Path = "depth-anything/Depth-Anything-V2-base-hf",
                 device=torch.device("cuda:0"), offload_from_device: bool = True):
        self.offload_from_device = offload_from_device
        self.device = device
        if self.offload_from_device:
            self.pipe = tr_pipeline("depth-estimation", model=checkpoint)
        else:
            self.pipe = tr_pipeline("depth-estimation", model=checkpoint, device=self.device)

    def run(self, pil_ims: List[Image.Image]) -> list[dict]:
        if self.offload_from_device:
            self.pipe.device = self.device
            self.pipe.model.to(self.device)
        depth_output = self.pipe(pil_ims, device=self.device)
        if self.offload_from_device:
            self.pipe.device = torch.device("cpu")
            self.pipe.model.to(self.pipe.device)

        return depth_output


def get_vals_in_front_of_obj(img_mask: np.ndarray, frame_monodepth_results: dict | np.ndarray,
                             grey_out_gt_mask: bool = True) -> np.ndarray:
    if isinstance(frame_monodepth_results, dict):
        depth_img_arr = np.array(frame_monodepth_results["depth"])
    else:
        depth_img_arr = frame_monodepth_results
    depth_vals_for_object = depth_img_arr[img_mask == 1]
    in_front_obj = np.zeros_like(depth_img_arr)
    if len(depth_vals_for_object):
        avg_depth_val = depth_vals_for_object.mean()
        in_front_obj[depth_img_arr > avg_depth_val] = 255
    if grey_out_gt_mask:
        in_front_obj[img_mask == 1] = 127

    return in_front_obj

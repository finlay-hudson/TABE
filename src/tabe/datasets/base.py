import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.tabe.configs.runtime_config import DataConfigBase


class Dataset:
    def __init__(self, data_cfg: DataConfigBase, ds_root: Optional[Path] = None):
        self.ds_root = ds_root
        if self.ds_root is None:
            self.ds_root = data_cfg.data_root
        assert self.ds_root.exists(), f"Dataset root: {self.ds_root} does not exists"
        self.frame_dir_name = data_cfg.frame_dir_name
        self.gt_mask_dir_name = data_cfg.gt_mask_dir_name
        self.vis_mask_dir_name = data_cfg.vis_mask_dir_name
        self.anno_file_name = data_cfg.anno_file_name

    def get_data_for_vid(self, vid_name: str) -> (list[Image.Image], Optional[np.ndarray], np.ndarray, Optional[list]):
        vid_dir = self.ds_root / vid_name
        anno = None
        if self.anno_file_name:
            with open(vid_dir / (self.anno_file_name + ".json")) as f:
                anno = json.load(f)
        file_names = sorted(list((vid_dir / self.frame_dir_name).glob("*.jpg")))
        all_ims_pil = [Image.open(file_names[frame_idx]) for frame_idx in range(len(file_names))]
        gt_masks = None
        if self.gt_mask_dir_name:
            gt_masks = np.stack([Image.open(m) for m in sorted(list((vid_dir / self.gt_mask_dir_name).glob("*.png")))])
        vis_masks = np.stack([Image.open(m) for m in sorted(list((vid_dir / self.vis_mask_dir_name).glob("*.png")))])

        return all_ims_pil, gt_masks, vis_masks, anno

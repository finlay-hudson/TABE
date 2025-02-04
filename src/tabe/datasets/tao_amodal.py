"""
TAO-Amodal Datasets expected to be in the structure:

TAO-Amodal/
├── frames/
├── amodal_annotations/
├── BURST_annotations/
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.tabe.configs.runtime_config import DataConfigTAOAmodal

import pycocotools.mask as cocomask

from src.tabe.datasets.base import Dataset
from src.tabe.datasets.utils import DatasetTypes


def rle_to_numpy(rle, shape):
    """
    Convert RLE (Run-Length Encoding) to a binary NumPy mask array.

    Parameters:
        rle (list or str): The RLE, as a list of alternating starts and lengths or a space-separated string.
        shape (tuple): The desired shape of the output mask (height, width).

    Returns:
        np.ndarray: A binary mask array with the given shape.
    """
    # If RLE is a string, convert it to a list of integers
    if isinstance(rle, str):
        rle = list(map(int, rle.split()))

    # Create a flat array initialized with zeros
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Decode RLE
    for start, length in zip(rle[::2], rle[1::2]):
        start -= 1  # Convert to zero-indexed
        mask[start: start + length] = 1

    # Reshape to the desired shape
    return mask.reshape(shape)


class TAOAmodalDataset(Dataset):
    def __init__(self, data_cfg: DataConfigTAOAmodal, ds_root: Optional[Path] = None):
        super().__init__(data_cfg, ds_root)
        self.track_idx = data_cfg.track_idx
        self.name = DatasetTypes.TAOAMODAL.name

    def get_data_for_vid(self, vid_name: str):
        frame_dir = self.ds_root / self.frame_dir_name / vid_name
        assert frame_dir.exists(), f"Frame dir: {frame_dir} does not exists"
        gt_masks = None
        if vid_name.split("/")[0] == "train":
            vid_mask_anno_fn = self.ds_root / self.vis_mask_dir_name / "train/train_visibility.json"
        else:
            vid_mask_anno_fn = self.ds_root / self.vis_mask_dir_name / f"{vid_name.split('/')[0]}/all_classes_visibility.json"
        with open(vid_mask_anno_fn) as f:
            vid_mask_annos = json.load(f)

        all_ims_pil = None
        all_vis_masks = {}
        for v_anno in vid_mask_annos["sequences"]:
            if v_anno["seq_name"] == vid_name.split("/")[-1]:
                file_names = [frame_dir / vid_name for vid_name in v_anno["annotated_image_paths"]]
                all_ims_pil = [Image.open(file_names[frame_idx]) for frame_idx in range(len(file_names))]
                for frame_idx, frame_segmentations in enumerate(v_anno["segmentations"]):
                    for seg_idx, seg in frame_segmentations.items():
                        # Add to the corresponding segment
                        if seg_idx not in all_vis_masks:
                            all_vis_masks[seg_idx] = [None] * len(v_anno["segmentations"])  # Initialize with None
                        all_vis_masks[seg_idx][frame_idx] = seg

        assert all_ims_pil, "Could not get images"

        vis_masks = all_vis_masks[str(self.track_idx)]
        for i, frame_seg in enumerate(vis_masks):
            if frame_seg is None:
                vis_masks[i] = np.zeros(all_ims_pil[0].size[::-1]).astype(np.uint8)
            else:
                vis_masks[i] = cocomask.decode(
                    {"size": all_ims_pil[0].size[::-1], "counts": vis_masks[i]["rle"].encode("utf-8")}).astype(np.uint8)

        vis_masks = np.stack(vis_masks) * 255
        assert len(vis_masks), f"Could not get visible masks for idx {self.track_idx}"

        return all_ims_pil, gt_masks, vis_masks, None

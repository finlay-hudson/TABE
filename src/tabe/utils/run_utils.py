from omegaconf import OmegaConf
from typing import Optional

import numpy as np
from PIL import ImageOps
import torch

from src.tabe.configs.runtime_config import RuntimeConfig
from src.tabe.datasets.factory import get_gt_data_cls
from src.tabe.datasets.tabe import get_vid_names
from src.tabe.datasets.utils import DatasetTypes


def setup(exp_name: Optional[str] = None, just_eval: bool = False):
    cfg = OmegaConf.structured(RuntimeConfig())
    if exp_name is not None:
        cfg.exp_name = exp_name
    torch.manual_seed(cfg["seed"])

    ds_type = cfg.dataset
    if ds_type == DatasetTypes.TABE51:
        vid_names = get_vid_names()
        if len(cfg.video_names):
            for vid_name in cfg.video_names:
                assert vid_name in vid_names, f"{vid_name} not in {vid_names} for TABE-51 dataset"
            vid_names = cfg.video_names
    else:
        assert len(cfg.video_names) > 0, "For a custom dataset, must have provided at least one video name"
        vid_names = cfg.video_names

    ds_cls = None
    if not just_eval:
        # get the data per dataset type
        ds_cls = get_gt_data_cls(ds_type, cfg.data)

    return cfg, ds_cls, vid_names


def pad_inputs(all_ims_pil, vis_masks, np_ims, gt_amodal_masks, monodepth_results, pad=150):
    all_ims_pil = [ImageOps.expand(im, border=pad, fill=(255, 255, 255)) for im in all_ims_pil]
    vis_masks = np.pad(vis_masks, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    np_ims = np.pad(np_ims, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=255)
    if gt_amodal_masks is not None:
        gt_amodal_masks = np.pad(gt_amodal_masks, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    for md_res in monodepth_results:
        md_res["depth"] = ImageOps.expand(md_res["depth"], border=pad, fill=255)

    return all_ims_pil, vis_masks, np_ims, gt_amodal_masks, monodepth_results

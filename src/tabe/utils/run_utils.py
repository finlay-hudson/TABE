from omegaconf import OmegaConf
from typing import Optional

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
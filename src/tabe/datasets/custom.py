"""
Datasets expected to be in the structure:

video_name/
├── frames/          # All frames of the video in numerical ordere
├── visible_masks/   # Visible masks in numerical order (must include at least the query mask)
├── gt_masks/        # (Optional) Ground truth amodal masks for frames
└── annos.json       # (Optional) File with a dict of {"occlusion": [occlusion level strings, mapped to OcclusionLevel]}

Example of this structure shown in: examples/
"""

from pathlib import Path
from typing import Optional

from src.tabe.configs.runtime_config import DataConfigCustom
from src.tabe.datasets.utils import DatasetTypes
from src.tabe.datasets.base import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_cfg: DataConfigCustom, ds_root: Optional[Path] = None):
        super().__init__(data_cfg, ds_root)
        self.name = DatasetTypes.CUSTOM.name

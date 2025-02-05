from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.tabe.configs.video_diffusion_config import VideoDiffusionConfig
from src.tabe.datasets.utils import DatasetTypes


class DataConfigBase(ABC):
    data_root: Path = Path("")
    frame_dir_name: str = ""
    gt_mask_dir_name: str = ""
    vis_mask_dir_name: str = ""
    anno_file_name: str = ""


@dataclass
class DataConfigTABE(DataConfigBase):
    data_root: Path = Path("data/TABE-51/data")
    frame_dir_name: str = "frames"
    gt_mask_dir_name: str = "gt_masks"
    vis_mask_dir_name: str = "visible_masks"
    anno_file_name: str = "annos"


@dataclass
class DataConfigCustom(DataConfigBase):
    data_root: Path = Path("examples")
    frame_dir_name: str = "frames"
    gt_mask_dir_name: Optional[str] = None
    vis_mask_dir_name: str = "visible_masks"
    anno_file_name: Optional[str] = None


@dataclass
class DataConfigTAOAmodal(DataConfigBase):
    data_root: Path = Path("data/TAO-Amodal")
    frame_dir_name: str = "frames"
    gt_mask_dir_name: Optional[str] = None
    vis_mask_dir_name: str = "BURST_annotations"
    anno_file_name: str = "amodal_annotations"
    track_idx: int = 1


@dataclass
class DataConfig:
    tabe: DataConfigTABE = DataConfigTABE
    custom: DataConfigCustom = DataConfigCustom
    tao_amodal: DataConfigTAOAmodal = DataConfigTAOAmodal
    output_root: Path = Path("outputs")


@dataclass
class RuntimeConfig:
    exp_name: str = "default"
    data: DataConfig = DataConfig
    # dataset: DatasetTypes = DatasetTypes.TABE51
    dataset: DatasetTypes = DatasetTypes.CUSTOM
    video_diffusion: VideoDiffusionConfig = VideoDiffusionConfig
    sam_checkpoint: Path = Path("checkpoints/sam2/sam2_hiera_large.pt")
    # Leaving this tuple empty, tuple(), for the TABE-51 dataset means all videos are ran
    video_names: tuple = tuple(["air_hockey_1"])
    use_gt_vis_mask: bool = False  # For TABE-51 dataset, for creating evaluation metrics this must be False
    use_gt_occlusion: bool = False  # For TABE-51 dataset, for creating evaluation metrics this must be False
    use_gt_bboxes: bool = False  # For TABE-51 dataset, for creating evaluation metrics this must be False
    or_orig_vis_mask: bool = True
    seed: int = 42
    added_bbox_perc: int = 0  # If there is any extension wanted on predicted bounding boxes

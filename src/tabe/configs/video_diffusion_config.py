from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VideoDiffusionNoiseSchedulerConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "linear"
    steps_offset: int = 1
    clip_sample: bool = False


@dataclass
class VideoDiffusionMotionModule:
    num_attention_heads: int = 8
    num_transformer_block: int = 1
    attention_block_types: tuple = (
    "Temporal_Self", "Temporal_Self", "Temporal_Light_down_resize", "Temporal_Text_Cross")
    temporal_position_encoding: bool = True
    temporal_position_encoding_max_len: int = 64
    temporal_attention_dim_div: int = 1
    text_cross_attention_dim: int = 768
    vision_cross_attention_dim: int = 768


@dataclass
class VideoDiffusionUnetConfig:
    motion_module_type: str = "Vanilla"
    motion_module_kwargs: VideoDiffusionMotionModule = VideoDiffusionMotionModule
    use_inflated_groupnorm: bool = True
    unet_use_cross_frame_attention: bool = False
    unet_use_temporal_attention: bool = False
    use_motion_module: bool = True
    motion_module_resolutions: tuple = tuple([1, 2, 4, 8])
    motion_module_mid_block: bool = True
    motion_module_decoder_only: bool = False


@dataclass
class AcceleratorConfig:
    mixed_precision: Optional[str] = None
    logging_dir: Optional[str] = None


@dataclass
class VideoDiffusionTrainingDataloaderConfig:
    consistent_transforms: bool = True
    use_white_background: bool = True
    train_batch_size: int = 1
    num_frames: int = 14
    size: int = 256


@dataclass
class VideoDiffusionTrainingConfig:
    accelerator: AcceleratorConfig = AcceleratorConfig()
    max_train_steps: int = 500
    gradient_accumulation_steps: int = 1
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    with_weightings: bool = True
    lr_scheduler_name: str = "constant"
    dataloader: VideoDiffusionTrainingDataloaderConfig = VideoDiffusionTrainingDataloaderConfig
    force_train: bool = False
    # This is set to not use too much memory, but if you wish to cache trained diffusion models, set to False
    no_cache: bool = True
    # Experimental feature, use shown in train_video_diffusion function
    fast_training_strategy: bool = False


@dataclass
class VideoDiffusionConfig:
    model_weight_dtype: str = "float32"
    resolution: int = 512
    prompt: str = "a video of sks"
    num_vids_to_generate: int = 5
    n_infer_steps: int = 50
    low_cpu_mem: bool = True
    sd_inpainting_model_path: Path = Path("checkpoints/stable-diffusion-v1-5-inpainting")
    cococo_unet_weights: Path = Path("checkpoints/cococo")
    seed: int = 42
    noise_scheduler_kwargs: VideoDiffusionNoiseSchedulerConfig = VideoDiffusionNoiseSchedulerConfig
    unet_additional_kwargs: VideoDiffusionUnetConfig = VideoDiffusionUnetConfig
    training: VideoDiffusionTrainingConfig = VideoDiffusionTrainingConfig

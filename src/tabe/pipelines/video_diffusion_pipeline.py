from pathlib import Path

from accelerate import Accelerator
from omegaconf import OmegaConf

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
import math
import numpy as np
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from src.tabe.configs.video_diffusion_config import VideoDiffusionConfig
from src.tabe.modules.components.monodepth import get_vals_in_front_of_obj
from src.tabe.modules.components.sam2 import get_sam2_predictor
from src.tabe.modules.components.video_diffusion_dataloader import create_video_dataloader
from src.tabe.modules.mask_from_gen_vid import create_masks_from_gen_vid
from src.tabe.modules.video_diffusion import make_video_diffusion_outpainting_inputs, run_video_diffusion_infer, \
    prepare_diffusion_train, train_video_diffusion
from src.tabe.utils.bbox_utils import draw_psuedo_bbox_on_input
from src.tabe.utils.occlusion_utils import OcclusionLevel, OcclusionInfo
from src.tabe.utils.torch_utils import str_to_torch_dtype
from third_party.COCOCO.cococo.models.unet import UNet3DConditionModel
from third_party.COCOCO.cococo.pipelines.pipeline_animation_inpainting_cross_attention_vae import \
    AnimationInpaintPipeline


class VideoDiffusionPipeline:
    def __init__(self, cfg: VideoDiffusionConfig, device: torch.device, model_out_dir: Path):
        self.cfg = cfg
        self.device = device
        self.model_weight_dtype = str_to_torch_dtype(self.cfg["model_weight_dtype"])
        self._load_components()
        self.model_out_dir = model_out_dir

    def _load_components(self):
        # For now we just load the components onto the cpu so we can put on gpu when required
        self.noise_scheduler = DDIMScheduler(**OmegaConf.to_container(self.cfg["noise_scheduler_kwargs"]))
        self.vae = AutoencoderKL.from_pretrained(self.cfg.sd_inpainting_model_path, subfolder="vae",
                                                 use_safetensors=False)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.sd_inpainting_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.cfg.sd_inpainting_model_path, subfolder="text_encoder")
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            self.cfg.sd_inpainting_model_path, subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(self.cfg.unet_additional_kwargs)
        )
        self.vae = self.vae.half().eval()
        self.text_encoder = self.text_encoder.half().eval()
        self.unet = self.unet.half().eval()

    def _load_cococo_weights(self, cococo_unet_weights_path: Path, device: torch.device) -> None:
        state_dict = {}
        for i in range(4):
            state_dict2 = torch.load(cococo_unet_weights_path / f'model_{i}.pth', map_location=device)
            state_dict = {**state_dict, **state_dict2}

        state_dict2 = {}
        for key in state_dict:
            if 'pe' in key:
                continue
            state_dict2[key.split('module.')[1]] = state_dict[key]

        self.unet.load_state_dict(state_dict2, strict=False)

    def _create_infer_inputs(self, ims, vis_masks, occlusion_info, estimated_bboxes, monodepth_results,
                             min_mask_points=250, added_bbox_perc=0, img_padding=False):
        assert len(ims) == len(vis_masks) == len(occlusion_info) == len(estimated_bboxes) == len(monodepth_results)
        input_ims, input_masks = [], []
        for im, v_mask, occl_info, e_bbox, md in zip(ims, vis_masks, occlusion_info, estimated_bboxes,
                                                     monodepth_results):
            if v_mask.max() > 1:
                v_mask = v_mask // 255
            in_front_obj_arr = get_vals_in_front_of_obj(v_mask, md)
            in_front_obj_and_psuedo_bbox, bounding_box_mask = draw_psuedo_bbox_on_input(im.size[::-1],
                                                                                        e_bbox,
                                                                                        in_front_obj_arr,
                                                                                        added_bbox_perc)
            input_img, input_mask = make_video_diffusion_outpainting_inputs(bounding_box_mask, im, v_mask,
                                                                            in_front_obj_and_psuedo_bbox,
                                                                            resolution=self.cfg["resolution"],
                                                                            min_mask_points=min_mask_points)
            # we dont want outpainting on non occluded frames unless there is image padding
            no_gens = {OcclusionLevel.NO_OCCLUSION, OcclusionLevel.NONE} if not img_padding else {
                OcclusionLevel.NO_OCCLUSION}
            if occl_info in no_gens:
                input_mask = Image.fromarray(np.zeros_like(input_mask))
            input_ims.append(input_img)
            input_masks.append(input_mask)

        return input_ims, input_masks

    def run_infer(self, ims, vis_masks, occlusion_info, estimated_bboxes, monodepth_results, sam_checkpoint,
                  image_padding=False):
        input_ims, input_masks = self._create_infer_inputs(ims, vis_masks, occlusion_info, estimated_bboxes,
                                                           monodepth_results, img_padding=image_padding)
        infer_pipe = AnimationInpaintPipeline(self.vae, self.text_encoder, self.tokenizer, self.unet,
                                              self.noise_scheduler)
        infer_pipe.to(self.device)
        all_generated_ims = run_video_diffusion_infer(infer_pipe, input_ims, input_masks, self.device,
                                                      dtype=torch.float16, guidance_scale=20, prompt=self.cfg["prompt"],
                                                      n_rounds=self.cfg["num_vids_to_generate"],
                                                      n_steps=self.cfg["n_infer_steps"], generator=None,
                                                      seed=self.cfg["seed"], resolution=self.cfg["resolution"])
        infer_pipe.to("cpu")
        all_gen_masks = []
        sam_pred = get_sam2_predictor(checkpoint=sam_checkpoint, device=self.device)
        for i, vid in enumerate(all_generated_ims):
            # sam_pred = get_sam2_predictor(checkpoint=sam_checkpoint, device=self.device)
            all_gen_masks.append(create_masks_from_gen_vid(vid, vis_masks, ims, occlusion_info, sam_pred=sam_pred,
                                                           resize_to=ims[0].size))

        return all_generated_ims, all_gen_masks, input_ims, input_masks

    def train_components(self, images: list[Image.Image], masks: np.ndarray, occl_info: list[OcclusionInfo]) -> None:
        train_settings = self.cfg["training"]

        if self.model_out_dir.exists() and not train_settings["force_train"]:
            print("Loading model")
            self.unet = UNet3DConditionModel.from_pretrained(self.model_out_dir, subfolder="unet",
                                                             low_cpu_mem_usage=self.cfg["low_cpu_mem"],
                                                             torch_dtype=self.model_weight_dtype)
            self.text_encoder = CLIPTextModel.from_pretrained(self.model_out_dir, subfolder="text_encoder",
                                                              low_cpu_mem_usage=self.cfg["low_cpu_mem"],
                                                              torch_dtype=self.model_weight_dtype)
            print("Model loaded")
        else:
            self._load_cococo_weights(self.cfg["cococo_unet_weights"], device=self.device)
            accelerator = Accelerator(
                gradient_accumulation_steps=train_settings["gradient_accumulation_steps"],
                mixed_precision=train_settings["accelerator"]["mixed_precision"],
                log_with=None,
                project_dir=train_settings["accelerator"]["logging_dir"],
            )

            # Dataset and DataLoaders creation:
            train_ds, train_dl = create_video_dataloader(images, masks, occl_info, self.tokenizer,
                                                         train_settings, self.cfg["prompt"])

            lr_scheduler, optimizer, text_encoder, unet = prepare_diffusion_train(accelerator, self.text_encoder,
                                                                                  train_settings, self.unet, self.vae)

            num_update_steps_per_epoch = math.ceil(len(train_dl) / train_settings["gradient_accumulation_steps"])
            lr_scheduler = get_scheduler(train_settings["lr_scheduler_name"], optimizer=optimizer,
                                         num_warmup_steps=train_settings["lr_warmup_steps"] *
                                                          train_settings["gradient_accumulation_steps"],
                                         num_training_steps=train_settings["max_train_steps"] *
                                                            train_settings["gradient_accumulation_steps"],
                                         num_cycles=train_settings["lr_num_cycles"], power=train_settings["lr_power"])
            unet, text_encoder, optimizer, train_dl = accelerator.prepare(unet, text_encoder, optimizer, train_dl)
            num_train_epochs = math.ceil(train_settings["max_train_steps"] / num_update_steps_per_epoch)
            unet, text_encoder, losses, _ = train_video_diffusion(train_dl, accelerator, self.vae, unet,
                                                                  text_encoder, lr_scheduler,
                                                                  self.noise_scheduler,
                                                                  train_settings["max_train_steps"],
                                                                  optimizer, num_train_epochs,
                                                                  self.model_weight_dtype,
                                                                  with_weightings=train_settings[
                                                                      "with_weightings"],
                                                                  fast_training_strategy=train_settings[
                                                                      "fast_training_strategy"])
            del optimizer, train_dl, lr_scheduler
            unet.to(self.model_weight_dtype)
            text_encoder.to(self.model_weight_dtype)
            if not train_settings["no_cache"]:
                print("Saving model")
                self.model_out_dir.mkdir(exist_ok=True, parents=True)
                trained_vid_unet = accelerator.unwrap_model(unet).merge_and_unload()
                trained_vid_text_encoder = accelerator.unwrap_model(text_encoder).merge_and_unload()
                trained_vid_unet.save_pretrained(self.model_out_dir / "unet")
                trained_vid_text_encoder.save_pretrained(self.model_out_dir / "text_encoder")
                print(f"Model saved to {self.model_out_dir}")
            self.unet = unet
            self.text_encoder = text_encoder

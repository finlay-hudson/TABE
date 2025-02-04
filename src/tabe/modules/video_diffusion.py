from einops import rearrange
import gc
import itertools

from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.tabe.utils.mask_utils import scale_up_mask
from third_party.COCOCO.cococo.pipelines.pipeline_animation_inpainting_cross_attention_vae import \
    AnimationInpaintPipeline


def prepare_data_for_video_diffusion_inference(imgs: np.ndarray | list, masks: np.ndarray | list, vae: AutoencoderKL,
                                               device: torch.device, dtype: torch.dtype):
    if isinstance(imgs, list):
        imgs = np.stack(imgs)
    if isinstance(masks, list):
        masks = np.stack(masks)
    if masks.max() > 1:
        masks = masks // 255
    if masks.ndim != imgs.ndim:
        masks = masks[..., None]

    imgs = 2 * (imgs / 255.0 - 0.5)

    pixel_values = torch.tensor(imgs).to(device=device, dtype=dtype).permute(0, 3, 1, 2)
    test_masks = torch.tensor(masks).to(device=device, dtype=dtype).permute(0, 3, 1, 2)

    masked_image = (1 - test_masks) * pixel_values
    vae.to(dtype)
    latents = vae.encode(masked_image).latent_dist.sample()
    test_masks = torch.nn.functional.interpolate(test_masks, size=latents.shape[-2:]).cuda()

    latents = rearrange(latents, "f c h w -> c f h w")
    test_masks = rearrange(test_masks, "f c h w -> c f h w")

    latents = latents * 0.18215

    return latents[None, ...].to(device), test_masks[None, ...].to(device)


@torch.no_grad()
def run_video_diffusion_infer(infer_pipe: AnimationInpaintPipeline, imgs: list | np.ndarray, masks: list | np.ndarray,
                              device: torch.device, dtype: torch.dtype = torch.float16,
                              n_rounds: int = 10, n_steps: int = 100, guidance_scale: int = 20,
                              prompt="a video of sks", generator=None, seed: int = 42, resolution: int = 512):
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(seed)

    infer_pipe.to(device).to(dtype)

    latents, test_masks = prepare_data_for_video_diffusion_inference(imgs, masks, infer_pipe.vae, device, dtype=dtype)

    all_vis_videos = []
    for step in range(n_rounds):
        with torch.no_grad():
            videos, masked_videos, recon_videos = infer_pipe(
                prompt,
                image=latents,
                masked_image=latents,
                masked_latents=None,
                masks=test_masks,
                generator=generator,
                video_length=latents.shape[2],
                negative_prompt=None,
                height=resolution,
                width=resolution,
                num_inference_steps=n_steps,
                unet_device=device,
                guidance_scale=guidance_scale
            )
        vis_videos = videos.permute(0, 2, 1, 3, 4).contiguous() / 0.18215

        images = []
        for i in range(len(vis_videos[0])):
            image = infer_pipe.vae.decode(vis_videos[0][i:i + 1].half()).sample
            images.append(image)
        vis_videos = torch.cat(images, dim=0)
        vis_videos = vis_videos / 2 + 0.5
        video = torch.clamp(vis_videos, 0, 1)
        vis_videos = video.permute(0, 2, 3, 1)

        all_vis_videos.append((255.0 * vis_videos.cpu().detach().numpy()).astype(np.uint8))

    infer_pipe.to("cpu")
    torch.cuda.empty_cache()

    return all_vis_videos


def make_video_diffusion_outpainting_inputs(bounding_box_mask, pil_im, gt_mask, in_front_obj_and_psuedo_bbox=None,
                                            resolution=512, min_mask_points=1000):
    assert bounding_box_mask.sum() > 0, "Bounding box mask is empty"
    bounding_box_mask = np.array(Image.fromarray(bounding_box_mask).resize((resolution, resolution), Image.NEAREST))
    if gt_mask.sum() == 0:
        return Image.fromarray(np.ones((resolution, resolution, 3), dtype=np.uint8) * 255), Image.fromarray(
            (bounding_box_mask * 255).astype(np.uint8))
    if isinstance(gt_mask, np.ndarray):
        gt_mask = Image.fromarray(gt_mask)
    reshaped_gt_mask = np.array(gt_mask.resize((resolution, resolution))).astype(np.uint8)
    if reshaped_gt_mask.sum() > 0:  # The resize can sometimes remove the mask
        while reshaped_gt_mask.sum() < min_mask_points:
            # if init mask is too small it doesn't generate anything
            reshaped_gt_mask = scale_up_mask(reshaped_gt_mask, iters=1, kernel_size=3)
    else:
        reshaped_gt_mask = bounding_box_mask

    if in_front_obj_and_psuedo_bbox is not None:
        in_front_obj_and_psuedo_bbox = np.array(
            Image.fromarray(in_front_obj_and_psuedo_bbox).resize((resolution, resolution), Image.NEAREST))
    pil_im = pil_im.resize((resolution, resolution))

    realfill_mask = 1 - reshaped_gt_mask
    realfill_img = np.array(pil_im).copy()
    realfill_img[realfill_mask == 1] = (255, 255, 255)
    realfill_mask *= 255
    realfill_mask_for_just_psuedo_bbox = realfill_mask.copy()
    realfill_mask_for_just_psuedo_bbox[bounding_box_mask == 0] = 0
    if in_front_obj_and_psuedo_bbox is not None:
        return Image.fromarray(realfill_img), Image.fromarray(in_front_obj_and_psuedo_bbox)

    return Image.fromarray(realfill_img), Image.fromarray(realfill_mask_for_just_psuedo_bbox)


def prepare_diffusion_train(accelerator, text_encoder, train_settings, unet, vae, lora_rank=8, lora_alpha=16,
                            lora_dropout=0.1, lora_bias='none'):
    unet_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                             lora_dropout=lora_dropout, bias=lora_bias)
    unet = get_peft_model(unet, unet_config)
    text_encoder_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha,
                                     target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
                                     lora_dropout=lora_dropout, bias=lora_bias)
    text_encoder = get_peft_model(text_encoder, text_encoder_config)
    vae.requires_grad_(False)
    # Optimizer creation
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class([
        {"params": unet.parameters(), "lr": 0.0002},
        {"params": text_encoder.parameters(), "lr": 4e-05}
    ], betas=(0.9, 0.999), weight_decay=0.01, eps=1e-08)
    unet, text_encoder, optimizer = accelerator.prepare(unet, text_encoder, optimizer)
    lr_scheduler = get_scheduler(train_settings["lr_scheduler_name"], optimizer=optimizer,
                                 num_warmup_steps=train_settings["lr_warmup_steps"] *
                                                  train_settings["gradient_accumulation_steps"],
                                 num_training_steps=train_settings["max_train_steps"] *
                                                    train_settings["gradient_accumulation_steps"],
                                 num_cycles=train_settings["lr_num_cycles"], power=train_settings["lr_power"])
    return lr_scheduler, optimizer, text_encoder, unet


def latent_sample(means, stds) -> torch.Tensor:
    from diffusers.utils.torch_utils import randn_tensor
    # make sure sample is on the same device as the parameters and has same dtype
    sample = randn_tensor(means.shape, generator=None, device=means.device, dtype=means.dtype)
    return means + stds * sample


def train_video_diffusion(train_dl, accelerator, vae, unet, text_encoder, lr_scheduler, noise_scheduler,
                          max_train_steps, optimizer, num_epochs, weight_dtype, initial_global_step=0, first_epoch=0,
                          with_weightings=False, fast_training_strategy=True):
    vae.to(unet.device, dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)
    unet.to(dtype=weight_dtype)
    progress_bar = tqdm(range(0, num_epochs), initial=initial_global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process, position=0, leave=True)
    global_step = initial_global_step
    all_losses = []
    model_preds = []
    frozen_elements = [[]] * len(train_dl.dataset)

    if fast_training_strategy:
        """
        vae encoding per batch is inefficient - this can be computed in advance. However, still left as a flag 
        defaulted to False as seems to produce ever so slightly worse results
        """
        for batch in train_dl:
            if batch["images"].ndim == 5:
                # Remove the batch dim
                batch["images"] = batch["images"][0]
                batch["conditioning_images"] = batch["conditioning_images"][0]
                batch["masks"] = batch["masks"][0]
                batch["weightings"] = batch["weightings"][0]
                batch["prompt_ids"] = batch["prompt_ids"][0]
            batch["images"].to(unet.device)
            batch["conditioning_images"].to(unet.device)
            batch["masks"].to(unet.device)
            batch["weightings"].to(unet.device)
            batch["prompt_ids"].to(unet.device)
            with torch.no_grad():
                latent_dists = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist

                # Convert masked images to latent space
                conditionings_latent_dists = vae.encode(batch["conditioning_images"].to(dtype=weight_dtype)).latent_dist

            # Downsample mask and weighting so that they match with the latents
            masks, size = batch["masks"].to(dtype=weight_dtype), latent_dists.mean.shape[2:]
            masks = F.interpolate(masks, size=size)

            weightings = batch["weightings"].to(dtype=weight_dtype)
            weightings = F.interpolate(weightings, size=size)

            frozen_elements[batch["index"].item()] = {"latents_means": latent_dists.mean,
                                                      "latents_stds": latent_dists.std,
                                                      "conditionings_means": conditionings_latent_dists.mean,
                                                      "conditionings_stds": conditionings_latent_dists.std,
                                                      "masks": masks, "weightings": weightings}
    for epoch in range(first_epoch, num_epochs):
        unet.train()
        text_encoder.train()

        epoch_losses = []
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(unet, text_encoder):
                if fast_training_strategy:
                    latents_means = torch.stack([frozen_elements[idx]["latents_means"] for idx in batch["index"]])
                    latents_stds = torch.stack([frozen_elements[idx]["latents_stds"] for idx in batch["index"]])
                    conditionings_means = torch.stack(
                        [frozen_elements[idx]["conditionings_means"] for idx in batch["index"]])
                    conditionings_stds = torch.stack(
                        [frozen_elements[idx]["conditionings_stds"] for idx in batch["index"]])
                    latents = latent_sample(latents_means, latents_stds) * 0.18215
                    conditionings_latent_dists = latent_sample(conditionings_means, conditionings_stds) * 0.18215
                    masks = torch.stack([frozen_elements[idx]["masks"] for idx in batch["index"]])
                    weightings = torch.stack([frozen_elements[idx]["weightings"] for idx in batch["index"]])

                    latents = rearrange(latents, "b f c h w -> b c f h w")
                    conditionings_latent_dists = rearrange(conditionings_latent_dists, "b f c h w -> b c f h w")
                    masks = rearrange(masks, "b f c h w -> b c f h w")
                    weightings = rearrange(weightings, "b f c h w -> b c f h w")
                else:
                    if batch["images"].ndim == 5:
                        # Remove the batch dim
                        batch["images"] = batch["images"][0]
                        batch["conditioning_images"] = batch["conditioning_images"][0]
                        batch["masks"] = batch["masks"][0]
                        batch["weightings"] = batch["weightings"][0]
                        batch["prompt_ids"] = batch["prompt_ids"][0]
                    batch["images"].to(unet.device)
                    batch["conditioning_images"].to(unet.device)
                    batch["masks"].to(unet.device)
                    batch["weightings"].to(unet.device)
                    batch["prompt_ids"].to(unet.device)

                    with torch.no_grad():
                        latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                        conditionings_latent_dists = vae.encode(
                            batch["conditioning_images"].to(dtype=weight_dtype)).latent_dist.sample()
                    test_masks = torch.nn.functional.interpolate(batch["masks"], size=latents.shape[-2:]).cuda()

                    latents = rearrange(latents, "f c h w -> c f h w")
                    conditionings_latent_dists = rearrange(conditionings_latent_dists, "f c h w -> c f h w")
                    test_masks = rearrange(test_masks, "f c h w -> c f h w")

                    latents = latents * 0.18215
                    conditionings_latent_dists = conditionings_latent_dists * 0.18215

                    latents = latents[None, ...]
                    conditionings_latent_dists = conditionings_latent_dists[None, ...]
                    masks = test_masks[None, ...]

                    weightings = batch["weightings"].to(dtype=weight_dtype)
                    weightings = F.interpolate(weightings, size=latents.shape[-2:])
                    weightings = rearrange(weightings, "f c h w -> c f h w")
                    weightings = weightings[None]

                bsz = len(latents)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                model_pred = unet(noisy_latents, conditionings_latent_dists, masks, timesteps,
                                  encoder_hidden_states=encoder_hidden_states,
                                  vision_encoder_hidden_states=None,
                                  device=unet.device).sample.to(dtype=latents.dtype)

                # Compute the diffusion loss
                if with_weightings:
                    loss = (weightings * F.mse_loss(model_pred.float(), noise.float(), reduction="none")).mean()
                else:
                    loss = (
                        F.mse_loss(model_pred.float(), noise.to(model_pred.device).float(), reduction="none")).mean()
                epoch_losses.append(loss.detach().cpu().numpy())

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)

                if accelerator.sync_gradients:
                    global_step += 1

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=False)

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)
                model_preds.append(model_pred.detach().cpu())

                if global_step >= max_train_steps:
                    break

        progress_bar.update(1)
        all_losses.append(np.mean(epoch_losses))

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.clear()
    accelerator.free_memory()

    del batch, accelerator, latents, loss, masks, noise, noisy_latents, weightings

    torch.cuda.empty_cache()
    gc.collect()

    return unet, text_encoder, all_losses, model_preds

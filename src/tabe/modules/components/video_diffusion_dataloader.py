import random
from typing import List

import numpy as np
from PIL import Image
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms_v2
from transformers import CLIPTokenizer

from src.tabe.utils.occlusion_utils import OcclusionInfo, OcclusionLevel


def make_mask_within_area(images: torch.Tensor, resolution: int, area_to_sample, times=30, min_scale=0.03,
                          max_scale=0.25) -> torch.Tensor:
    mask, times = torch.ones_like(images[0:1, :, :]), np.random.randint(1, times)
    min_size, max_size, margin = np.array([min_scale, max_scale, 0.01]) * resolution
    max_size = min(max_size, resolution - margin * 2)

    for _ in range(times):
        width = np.random.randint(int(min_size), int(max_size))
        height = np.random.randint(int(min_size), int(max_size))

        # Find valid positions in `area_to_sample` where area_to_sample greater than 0
        valid_y, valid_x = np.where(area_to_sample > 0)

        # If there are valid points, select a random starting point from valid locations
        if len(valid_y):
            idx = np.random.randint(0, len(valid_y))  # Random index within valid points
            y_start, x_start = valid_y[idx].item(), valid_x[idx].item()

            # Ensure the box is within image boundaries by adjusting for width and height
            if (y_start + height < resolution) and (x_start + width < resolution):
                mask[:, y_start:y_start + height, x_start:x_start + width] = 0

    mask = 1 - mask if random.random() < 0.5 else mask
    return mask


def make_mask(images: torch.Tensor, resolution: int, times: int = 30) -> torch.Tensor:
    mask, times = torch.ones_like(images[0:1, :, :]), np.random.randint(1, times)
    min_size, max_size, margin = np.array([0.03, 0.25, 0.01]) * resolution
    max_size = min(max_size, resolution - margin * 2)

    for _ in range(times):
        width = np.random.randint(int(min_size), int(max_size))
        height = np.random.randint(int(min_size), int(max_size))

        x_start = np.random.randint(int(margin), resolution - int(margin) - width + 1)
        y_start = np.random.randint(int(margin), resolution - int(margin) - height + 1)
        mask[:, y_start:y_start + height, x_start:x_start + width] = 0

    mask = 1 - mask if random.random() < 0.5 else mask
    return mask


class DiffusionVideoDataset(Dataset):
    """
    A dataset to prepare the training and conditioning images and
    the masks with the dummy prompt for fine-tuning the model.
    It pre-processes the images, masks and tokenizes the prompts.
    """

    def __init__(self, images: List[Image.Image], masks: np.ndarray, occlusion_info: List[OcclusionInfo], tokenizer,
                 train_prompt: str = "a photo of sks", size: int = 512, use_white_background: bool = False,
                 num_frames: int = 14, consistent_transforms: bool = False, masking_mode: str = "all"):
        self.size = size
        self.tokenizer = tokenizer

        self.masks = masks

        occlusion_levels = np.array([occl.level for occl in occlusion_info])
        amount_of_occlusion = np.array([occl.amount for occl in occlusion_info])

        self.train_prompt = train_prompt

        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RandomResize(size, int(1.125 * size)),
                transforms_v2.RandomCrop(size),
                transforms_v2.ToImage(),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize([0.5], [0.5]),  # between -1 and 1
            ]
        )
        self.use_white_background = use_white_background
        self.no_occlusion_idxs = np.where(np.array(occlusion_levels) == OcclusionLevel.NO_OCCLUSION)[0]
        min_examples = 1
        max_occl_amount = 0.1
        if len(self.no_occlusion_idxs) < min_examples:
            # If has less than certain amount of examples - add on the least occluded indicies as long as they are
            # occluded below a threshold
            lowest_idxs_of_occlusion = np.array([idx for idx in np.argsort(amount_of_occlusion) if
                                                 idx not in self.no_occlusion_idxs and occlusion_levels[
                                                     idx] != OcclusionLevel.NONE])
            lowest_idxs_of_occlusion = lowest_idxs_of_occlusion[
                amount_of_occlusion[lowest_idxs_of_occlusion] < max_occl_amount]
            self.no_occlusion_idxs = np.concatenate([self.no_occlusion_idxs, lowest_idxs_of_occlusion[:min_examples]])
        self.num_frames = num_frames
        self.consistent_transforms = consistent_transforms
        self.images = images
        self.masking_mode = masking_mode

    def _create_mask(self, image: torch.Tensor, resized_gt_mask: torch.Tensor):
        if self.masking_mode == "all":
            if random.random() < 0.1:
                mask = torch.ones_like(image[0:1])
            elif random.random() < 0.5:
                mask = 1 - torch.from_numpy(resized_gt_mask).unsqueeze(0).to(image.dtype)
            else:
                mask = make_mask_within_area(image, self.size, resized_gt_mask)
        elif self.masking_mode == "just_area":
            mask = make_mask_within_area(image, self.size, resized_gt_mask)
        else:
            if random.random() < 0.5:
                mask = make_mask(image, self.size)
            else:
                mask = make_mask_within_area(image, self.size, resized_gt_mask)
        return mask

    def __len__(self):
        return len(self.no_occlusion_idxs)

    def __getitem__(self, index: int):
        example = {"debug_ims": [], "debug_gt_masks": []}
        non_occluded_idx = self.no_occlusion_idxs[index]
        images, weightings, masks, not_occluded, gt_masks = [], [], [], [], []
        img_size = None
        for frame_idx in range(non_occluded_idx, non_occluded_idx + self.num_frames):
            if frame_idx < len(self.images):
                image = self.images[frame_idx]
                example["debug_ims"].append(np.array(image))
                gt_mask = self.masks[frame_idx]
                if gt_mask.max() > 1:
                    gt_mask = gt_mask // 255
                example["debug_gt_masks"].append(np.array(gt_mask))
                if self.use_white_background:
                    np_img = np.array(image)
                    np_img[gt_mask == 0] = (255, 255, 255)
                    image = Image.fromarray(np_img)
                img_size = image.size
            else:
                if img_size is None:
                    raise ValueError("Needed to load at least one image first")
                image = Image.new("RGB", img_size, (255, 255, 255))
                example["debug_ims"].append(np.array(image))
                gt_mask = np.ones(image.size[::-1], dtype=np.uint8)
                example["debug_gt_masks"].append(np.array(gt_mask))

            weighting = Image.fromarray(gt_mask)
            gt_masks.append(gt_mask)
            images.append(image)
            weightings.append(weighting)
            not_occluded.append(frame_idx in self.no_occlusion_idxs)

        transformed_inputs = self.transform(*images, *weightings)
        images = transformed_inputs[:len(images)]
        weightings = list(transformed_inputs[len(images):])

        for i, (image, weighting, gt_mask) in enumerate(zip(images, weightings, gt_masks)):
            resized_gt_mask = np.array(Image.fromarray(gt_mask).resize((self.size, self.size), Image.NEAREST))
            mask = self._create_mask(image, resized_gt_mask)
            masks.append(mask)
            weightings[i] = weighting[0:1] < 0

        example["images"] = torch.stack(images)
        example["weightings"] = torch.stack(weightings)
        example["masks"] = torch.stack(masks)
        example["conditioning_images"] = example["images"] * (example["masks"] < 0.5)
        example["debug_not_occluded"] = not_occluded
        example["weightings"][~np.array(not_occluded)] = 0

        train_prompt = "" if random.random() < 0.1 else self.train_prompt
        example["prompt_ids"] = self.tokenizer(
            train_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        example["index"] = index

        return example


def create_video_dataloader(images: List[Image.Image], masks: np.ndarray, occlusion_info: List[OcclusionLevel],
                            tokenizer: CLIPTokenizer, cfg: DictConfig, prompt: str) -> tuple[DiffusionVideoDataset,
                                                                                             DataLoader]:
    train_ds = DiffusionVideoDataset(images, masks, occlusion_info, tokenizer,
                                     num_frames=cfg["dataloader"]["num_frames"], size=cfg["dataloader"]["size"],
                                     use_white_background=cfg["dataloader"]["use_white_background"],
                                     train_prompt=prompt, consistent_transforms=True)

    return train_ds, DataLoader(train_ds, batch_size=cfg["dataloader"]["train_batch_size"], shuffle=True, num_workers=1)

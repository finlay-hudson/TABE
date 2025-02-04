import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from src.tabe.modules.components.sam2 import predict_masks_sam
from src.tabe.utils.occlusion_utils import OcclusionLevel


def convert_to_sam_vid_format(img: np.ndarray, img_size: int, n_max: int = 1):
    img = Image.fromarray(img).resize((img_size, img_size))
    img = pil_to_tensor(img)
    if n_max == 1:
        img = img / 255.0

    return img


def create_masks_from_gen_vid(gen_vid, vis_masks, frames, occlusion_info, sam_pred, resize_to=(512, 512)):
    query_mask = None
    running_frames = []
    for i, occl_info in enumerate(occlusion_info):
        if i == 0:
            query_mask = convert_to_sam_vid_format(np.array(vis_masks[i]), sam_pred.image_size)[0]
        if occl_info == OcclusionLevel.NO_OCCLUSION:
            white_bg_img = np.array(frames[i])
            white_bg_img[np.array(vis_masks[i]) == 0] = 255
        else:
            white_bg_img = gen_vid[i]
        running_frames.append(convert_to_sam_vid_format(white_bg_img, sam_pred.image_size))

    running_frames = torch.stack(running_frames)
    running_frames -= torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    running_frames /= torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

    gen_masks = predict_masks_sam(sam_pred, query_mask, running_frames,
                                  [sam_pred.image_size, sam_pred.image_size])

    for g_i, gen_mask in enumerate(gen_masks):
        if gen_mask.ndim == 3 and gen_mask.shape[0] == 1:
            gen_mask = gen_mask[0]
        gen_masks[g_i] = np.array(Image.fromarray(gen_mask).resize(resize_to, Image.NEAREST))

    return np.stack(gen_masks).astype(np.uint8)

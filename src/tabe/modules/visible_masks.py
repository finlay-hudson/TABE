from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from src.tabe.modules.components.sam2 import get_sam2_predictor, predict_masks_sam, sam_preprocessing
from src.tabe.utils.general_utils import suppress_output


def predict_visible_masks(sam_checkpoint: str | Path, frames: List[Image.Image], query_mask: np.ndarray) -> np.ndarray:
    orig_sizes = [f.size for f in frames]
    assert all(s == orig_sizes[0] for s in orig_sizes), "Frames must have the same size"

    sam_pred = get_sam2_predictor(checkpoint=sam_checkpoint)
    pp_frames, query_frames = sam_preprocessing(frames, Image.fromarray(query_mask),
                                                resize_to=(sam_pred.image_size, sam_pred.image_size))

    with suppress_output():  # Stop the tqdm
        pred_masks = predict_masks_sam(sam_pred, query_frames / 255., pp_frames, orig_sizes[0][::-1])
    # Unload memory
    pp_frames.detach().cpu()
    query_frames.detach().cpu()
    del pp_frames, query_frames, sam_pred

    # Set first predicted mask to the input query mask
    pred_masks[0] = query_mask / 255.
    pred_masks = (np.array(pred_masks) * 255).astype(np.uint8)

    return pred_masks

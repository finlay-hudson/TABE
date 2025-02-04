from natsort import natsorted
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor


def load_images_from_dir(img_dir, exclude_patterns=None, resize_to=None, device=None, out_max=255, img_mean=None,
                         img_std=None):
    """
    :param img_dir (str).
    :param exclude_patterns (Optional[list]) Patterns of filename wishing to be excluded from loading
    :param resize_to (Optional[list]) Size to resize images to
    :param device (Optional[torch.device]) Device to put images on
    :param out_max (int) max value for pixel on output, 255 would mean uint8 dtype and 1 would mean float32 dtype
    :param img_mean (Optional(list)) mean norm values
    :param img_std (Optional(list)) mean std values
    :return frames (T, H, W, 3) array with uint8 [0, 255].
    """
    if out_max not in [1, 255]:
        raise NotImplementedError(f"Do not have implementation for max value of {out_max}")
    img_dir = Path(img_dir)
    img_files = list(img_dir.glob("*"))
    remove_files = []
    if exclude_patterns is not None:
        if not (isinstance(exclude_patterns, list)):
            exclude_patterns = [exclude_patterns]
        for img_fp in img_files:
            for pattern in exclude_patterns:
                if pattern in img_fp.name:
                    remove_files.append(img_fp)
                    break

    img_files = natsorted(img_files)
    frames = []
    loaded_filepaths = []
    unique_img_shapes = set()
    orig_im_shapes = []
    for fp in img_files:
        if fp in remove_files:
            continue
        img_pil = Image.open(fp)
        orig_im_shapes.append(img_pil.size)
        if resize_to is not None:
            img_pil = img_pil.resize(resize_to)
        img = pil_to_tensor(img_pil)

        frames.append(img)
        unique_img_shapes.add(img_pil.size)
        loaded_filepaths.append(fp)

    if not len(frames):
        raise ValueError(f"No frames found in {img_dir}")
    if not len(unique_img_shapes) == 1:
        raise ValueError("More than one unique shape found in directory, reshaping needed")
    frames = torch.stack(frames)
    if out_max == 1:
        frames = frames / 255.0

    if img_mean is not None:
        # normalize by mean
        frames -= torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    if img_std is not None:
        # normalize by std
        frames /= torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if device is not None:
        frames = frames.to(device)

    return frames, loaded_filepaths, list(unique_img_shapes)[0], orig_im_shapes

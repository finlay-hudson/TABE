from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from src.tabe.pipelines.video_mask_gen_pipeline import GenerationOutputs
from src.tabe.utils.mask_utils import convert_one_to_three_channel
from src.tabe.utils.occlusion_utils import OcclusionInfo


class Visualiser:
    def __init__(self, vis_root_out_dir: Path):
        self.vis_root_out_dir: Path = vis_root_out_dir
        self.vis_root_out_dir.mkdir(exist_ok=True)
        self.bg_color: tuple = (255, 255, 255)
        self.gt_mask_col: tuple = (0, 255, 0)
        self.vis_mask_col: tuple = (0, 0, 255)
        self.pred_mask_col: tuple = (255, 0, 0)
        self.vis_root_out_dir_extra = self.vis_root_out_dir / "extra_vis"

    def _visualise_occlusion(self, ims: np.ndarray, occlusion_info: List[OcclusionInfo]) -> None:
        print("Making occlusion vis")
        out_dir_occlusion = self.vis_root_out_dir_extra / "occlusion_vis"
        out_dir_occlusion.mkdir(exist_ok=True)
        for frame_num, (im, occl) in enumerate(zip(ims, occlusion_info)):
            occl_vis = add_text(im.copy(), f"P: {occl.level.name}", bg_col=self.bg_color, scale=4, thickness=2)
            Image.fromarray(occl_vis).save(out_dir_occlusion / f"{frame_num}.jpg")

    def _visualise_estimated_bboxes(self, ims: np.ndarray, est_bboxes: np.ndarray) -> None:
        print("Making estimated bbox vis")
        out_dir_est_bbox = self.vis_root_out_dir_extra / "estimated_bbox_vis"
        out_dir_est_bbox.mkdir(exist_ok=True)
        for frame_num, (im, est_bbox) in enumerate(zip(ims, est_bboxes)):
            bbox_im = im.copy()
            if est_bbox is not None:
                cv2.rectangle(bbox_im, (est_bbox[1], est_bbox[0]), (est_bbox[3], est_bbox[2]), color=(255, 0, 0),
                              thickness=2)
            Image.fromarray(bbox_im).save(out_dir_est_bbox / f"{frame_num}.jpg")

    def _visualise_masks(self, ims: np.ndarray, vis_masks: np.ndarray, pred_amodal_masks: np.ndarray,
                         gt_amodal_masks: Optional[np.ndarray] = None, vid_num: int = 0) -> None:
        out_dir_masks = self.vis_root_out_dir_extra / "masks_vis" / str(vid_num)
        out_dir_masks.mkdir(exist_ok=True, parents=True)
        if gt_amodal_masks is None:
            gt_amodal_masks = [None] * len(ims)
        for frame_num, (im, vis_mask, pred_mask, gt_mask) in enumerate(
                zip(ims, vis_masks, pred_amodal_masks, gt_amodal_masks)):
            conc_ims = [im]
            if gt_mask is not None:
                gt_mask_im = overlay_mask_over_image(im.copy(), gt_mask.copy(), color=self.gt_mask_col)
                add_text(gt_mask_im, "GT Amodal Mask", bg_col=self.bg_color, scale=4, thickness=2)
                conc_ims.append(gt_mask_im)
            vis_mask_im = overlay_mask_over_image(im.copy(), vis_mask.copy(), color=self.vis_mask_col)
            add_text(vis_mask_im, "Visible Mask", bg_col=self.bg_color, scale=4, thickness=2)
            conc_ims.append(vis_mask_im)
            if not np.isnan(pred_mask).all():
                pred_mask_im = overlay_mask_over_image(im.copy(), pred_mask.copy(), color=self.pred_mask_col)
                add_text(pred_mask_im, "Pred Amodal Mask", bg_col=self.bg_color, scale=4, thickness=2)
                conc_ims.append(pred_mask_im)
            else:
                print(f"No pred mask for frame: {frame_num}")

            Image.fromarray(np.concatenate(conc_ims, axis=1)).save(out_dir_masks / f"{frame_num}.jpg")

    def _visualise_diffusion_model_inputs(self, input_ims: np.ndarray, input_masks: np.ndarray) -> None:
        print("Making diffusion model inputs vis")
        out_dir_diff_model_inputs = self.vis_root_out_dir_extra / "diffusion_model_inputs"
        out_dir_diff_model_inputs.mkdir(exist_ok=True)
        for frame_num, (input_im, input_mask) in enumerate(zip(input_ims, input_masks)):
            Image.fromarray(np.concatenate([input_im, convert_one_to_three_channel(input_mask)], axis=1)).save(
                out_dir_diff_model_inputs / f"{frame_num}.jpg")

    def _visualise_final_outputs(self, ims: np.ndarray, masks: np.ndarray, vid_num: int = 0) -> None:
        out_dir_final_outputs = self.vis_root_out_dir / "final_outputs" / str(vid_num)
        out_dir_final_outputs.mkdir(exist_ok=True, parents=True)
        out_dir_final_outputs_masks = out_dir_final_outputs / "masks"
        out_dir_final_outputs_masks.mkdir(exist_ok=True, parents=True)
        out_dir_final_outputs_vis_frames = out_dir_final_outputs / "vis_frames"
        out_dir_final_outputs_vis_frames.mkdir(exist_ok=True, parents=True)
        for frame_num, (im, mask) in enumerate(zip(ims, masks)):
            Image.fromarray(mask).save(out_dir_final_outputs_masks / f"{frame_num}.png")
            conc = np.concatenate([im, convert_one_to_three_channel(mask),
                                   overlay_mask_over_image(im, mask, color=self.pred_mask_col)])
            Image.fromarray(conc).save(out_dir_final_outputs_vis_frames / f"{frame_num}.jpg")

    def visualise(self, outputs: GenerationOutputs, gt_amodal_masks: Optional[np.ndarray] = None,
                  output_extra_vis: bool = True) -> None:
        n_vids_generated = outputs.masks.shape[0]
        print("Making final outputs vis")
        for vid_num in range(n_vids_generated):
            self._visualise_final_outputs(outputs.debugs.ims, outputs.masks[vid_num], vid_num)

        if output_extra_vis:
            self.vis_root_out_dir_extra.mkdir(exist_ok=True)
            self._visualise_occlusion(outputs.debugs.ims, outputs.debugs.occlusion)
            self._visualise_estimated_bboxes(outputs.debugs.ims, outputs.debugs.estimated_bboxes)
            self._visualise_diffusion_model_inputs(outputs.debugs.gen_input_ims.astype(np.uint8),
                                                   outputs.debugs.gen_input_masks.astype(np.uint8))

            for vid_num in range(n_vids_generated):
                self._visualise_masks(outputs.debugs.ims, outputs.debugs.vis_mask,
                                      outputs.debugs.pred_masks[vid_num], gt_amodal_masks, vid_num)


def overlay_mask_over_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5,
                            color: tuple = (0, 255, 0)) -> np.ndarray:
    # Ensure mask and image have the same dimensions
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask dimensions do not match.")
    if not mask.any():
        return image
    if mask.max() > 1:
        mask = mask.copy() // 255
    overlay_mask = np.zeros_like(image)
    overlay_mask[mask == 1] = color

    anno_img = image.copy()
    anno_img[mask == 1] = cv2.addWeighted(image[mask == 1], 1 - alpha, overlay_mask[mask == 1], alpha, 0)

    return anno_img


def add_text(frame: np.ndarray, txt: str, pos: tuple = (0, 0), font: int = cv2.FONT_HERSHEY_PLAIN, scale: float = 1.0,
             thickness: int = 1, txt_col: tuple = (0, 0, 0), bg_col: Optional[tuple] = None) -> np.ndarray:
    was_float = False
    if frame.dtype.kind == 'f':
        was_float = True
        frame = (frame * 255).astype(np.uint8)
    x, y = pos
    text_size, _ = cv2.getTextSize(txt, font, scale, thickness)
    text_w, text_h = text_size
    if bg_col is not None:
        cv2.rectangle(frame, (pos[0] - 5, pos[1] - 5), (x + text_w + 5, y + text_h + 5), bg_col, -1)
    cv2.putText(frame, txt, (x, int(y + text_h + 1)), font, scale, txt_col, thickness)

    if was_float:
        return frame / 255.0

    return frame

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image
import torch

from src.tabe.configs.runtime_config import RuntimeConfig
from src.tabe.modules.components.chunking import basic_video_chunker, chunk_postprocess
from src.tabe.modules.components.monodepth import MonoDepth
from src.tabe.modules.create_estimated_bbox import estimate_bboxes
from src.tabe.modules.visible_masks import predict_visible_masks
from src.tabe.modules.occlusion_predictor import predict_occlusion
from src.tabe.utils.occlusion_utils import map_occlusion_levels, OcclusionInfo, OcclusionLevel
from src.tabe.utils.bbox_utils import add_h_w_perc_to_bbox


@dataclass
class ExtraGenerationOutputs:
    ims: np.ndarray  # n_frames, h, w, 3
    pred_masks: np.ndarray  # n_gen_videos, n_frames, h, w, 3
    monodepth: np.ndarray  # n_frames, h, w
    vis_mask: np.ndarray  # n_frames, h, w
    occlusion: List[OcclusionInfo]  # n_frames
    estimated_bboxes: np.ndarray  # n_frames, 4
    gen_frames: np.ndarray  # n_gen_videos, n_frames, h, w, 3
    gen_input_ims: np.ndarray  # n_frames, res, res, 3
    gen_input_masks: np.ndarray  # n_frames, res, res


@dataclass
class GenerationOutputs:
    masks: np.ndarray  # n_gen_videos, n_frames, h, w
    debugs: ExtraGenerationOutputs


class VideoMaskGenerationPipeline:
    def __init__(self, cfg: RuntimeConfig, video_diffusion, device: torch.device = torch.device("cuda:0")):
        self.video_diffusion = video_diffusion
        self.monodepth = MonoDepth(device=device)
        self.cfg = cfg
        self.device = device

    def __call__(self, all_ims_pil: List[Image.Image], query_mask: np.ndarray,
                 gt_vis_masks: Optional[np.ndarray] = None, gt_occlusion: Optional[List[dict]] = None,
                 gt_amodal_masks: Optional[np.ndarray] = None):
        """
        :param all_ims_pil: list of pil frames, each of size (im_w, im_h)
        :param query_mask: shape(im_h, im_w)
        :param gt_vis_masks: optional(shape(n_frames, im_h, im_w))
        :param gt_occlusion: optional(shape(n_frames))
        :param gt_amodal_masks: optional(shape(n_frames, im_h, im_w))
        :return:
        """
        # Get the inputs required through the pipeline
        np_ims = np.stack([np.array(f) for f in all_ims_pil])
        monodepth_results = self.monodepth.run(all_ims_pil)
        vis_masks = self._get_pipeline_vis_masks(all_ims_pil, gt_vis_masks, query_mask)
        occlusion_info_with_amounts = self._get_pipeline_occlusion_levels(gt_occlusion, monodepth_results, np_ims,
                                                                          vis_masks)
        occlusion_levels = [occl.level for occl in occlusion_info_with_amounts]
        estimated_bboxes = self._get_pipeline_estimated_bboxes(gt_amodal_masks, monodepth_results, occlusion_levels,
                                                               vis_masks)

        # Train the diffusion model
        self.init_trained_models(all_ims_pil, vis_masks, occlusion_info_with_amounts)

        # Inference
        chunks = self._chunk_video_for_inference(occlusion_levels)
        all_gen_frames, all_input_ims, all_input_masks, all_pred_masks = self._create_empty_outputs(all_ims_pil)
        for chunk in chunks:
            if len(chunk) > 64:
                raise ValueError(
                    "Chunk too long, this is either caused by the occlusion info being wrong or the chunking logic or "
                    "the object is occluded too long. Currently only support clips of up to 64 frames")
            c_estimated_bboxes, c_ims, c_monodepth_results, c_occlusion, c_vis_masks = self._chunk_up_inputs(
                all_ims_pil, chunk, estimated_bboxes, monodepth_results, occlusion_levels, vis_masks)
            if np.all(np.array(c_occlusion) == OcclusionLevel.NO_OCCLUSION):
                for j, c in enumerate(chunk):
                    all_pred_masks[:, c] = c_vis_masks[j]
                continue

            print(f"Running generation for frames {chunk[0]}-{chunk[-1]}")
            gen_frames, gen_masks, input_ims, input_masks = self.video_diffusion.run_infer(c_ims, c_vis_masks,
                                                                                           c_occlusion,
                                                                                           c_estimated_bboxes,
                                                                                           c_monodepth_results,
                                                                                           self.cfg.sam_checkpoint)
            for i in range(len(gen_masks)):
                for j, c in enumerate(chunk):
                    all_pred_masks[i][c] = gen_masks[i][j]
                    if self.cfg["or_orig_vis_mask"]:
                        all_pred_masks[i][c] = np.maximum(all_pred_masks[i][c], c_vis_masks[j]).clip(0, 1)
                    all_gen_frames[i][c] = np.array(Image.fromarray(gen_frames[i][j]).resize(all_ims_pil[0].size))
                    if i == 0:
                        all_input_ims[c] = np.array(input_ims[j])
                        all_input_masks[c] = np.array(input_masks[j])

        self._output_checker(all_pred_masks, occlusion_levels)

        # Need to fill any missing pred masks with the visible mask
        final_amodal_masks = np.zeros_like(all_pred_masks).astype(np.uint8)
        for fn in range(all_pred_masks.shape[1]):
            if np.isnan(all_pred_masks[:, fn]).all():
                final_amodal_masks[:, fn] = vis_masks[fn]
            else:
                final_amodal_masks[:, fn] = (all_pred_masks[:, fn] * 255).astype(np.uint8)

        return GenerationOutputs(masks=final_amodal_masks,
                                 debugs=ExtraGenerationOutputs(
                                     ims=np_ims,
                                     pred_masks=all_pred_masks,
                                     monodepth=np.array([md["depth"] for md in monodepth_results]),
                                     vis_mask=vis_masks,
                                     occlusion=occlusion_info_with_amounts,
                                     estimated_bboxes=np.array(list(estimated_bboxes.values())),
                                     gen_frames=all_gen_frames,
                                     gen_input_ims=all_input_ims,
                                     gen_input_masks=all_input_masks,
                                 ))

    def _output_checker(self, all_pred_masks: np.ndarray, occlusion_info: List[OcclusionLevel]):
        if np.isnan(all_pred_masks[:, np.array(occlusion_info) != OcclusionLevel.NO_OCCLUSION]).any():
            raise ValueError(
                f"Something has gone wrong and indices {np.unique(np.where(np.isnan(all_pred_masks[:, np.array(occlusion_info) != OcclusionLevel.NO_OCCLUSION]).any(axis=(2, 3)))[1])} have not been generated")

    def _chunk_up_inputs(self, all_ims_pil: List[Image.Image], chunk: List[int], estimated_bboxes, monodepth_results,
                         occlusion_info, vis_masks):
        chunk_ims = [all_ims_pil[c] for c in chunk]
        chunk_vis_masks = [vis_masks[c] for c in chunk]
        chunk_occlusion = [occlusion_info[c] for c in chunk]
        chunk_estimated_bboxes = [estimated_bboxes[c] for c in chunk]
        chunk_monodepth_results = [monodepth_results[c] for c in chunk]

        return chunk_estimated_bboxes, chunk_ims, chunk_monodepth_results, chunk_occlusion, chunk_vis_masks

    def _create_empty_outputs(self, all_ims_pil: List[Image.Image]):
        res = self.cfg.video_diffusion.resolution
        out_im_shape = (self.cfg.video_diffusion.num_vids_to_generate, len(all_ims_pil), *all_ims_pil[0].size[::-1])
        all_pred_masks = np.ones(out_im_shape) * np.nan
        all_gen_frames = np.ones((*out_im_shape, 3)) * np.nan
        all_input_ims = np.ones((len(all_ims_pil), res, res, 3)) * np.nan
        all_input_masks = np.ones((len(all_ims_pil), res, res)) * np.nan
        return all_gen_frames, all_input_ims, all_input_masks, all_pred_masks

    @staticmethod
    def _chunk_video_for_inference(occlusion_info: List[OcclusionLevel], max_chunk_length: int = 35):
        pp_chunks = [[i for i in range(len(occlusion_info))]]
        if len(occlusion_info) > max_chunk_length:
            # We are best to split up the video into chunks
            chunks = basic_video_chunker(occlusion_info, min_frames=14, ideal_frames=30)
            pp_chunks = chunk_postprocess(chunks, occlusion_info, ideal_frames=30)

        return pp_chunks

    def _get_pipeline_estimated_bboxes(self, gt_amodal_masks: np.ndarray, monodepth_results: list[dict],
                                       occlusion_info: list[OcclusionLevel], vis_masks: np.ndarray):
        masks_for_estimating_bboxes = (
            gt_amodal_masks if self.cfg.use_gt_bboxes and gt_amodal_masks is not None else vis_masks)
        estimated_bboxes = estimate_bboxes(masks_for_estimating_bboxes, occlusion_info, monodepth_results)
        for fn in estimated_bboxes:
            estimated_bboxes[fn] = add_h_w_perc_to_bbox(estimated_bboxes[fn], self.cfg.added_bbox_perc,
                                                        self.cfg.added_bbox_perc)

        return estimated_bboxes

    def _get_pipeline_occlusion_levels(self, gt_occlusion: Optional[list[dict]], monodepth_results: list[dict],
                                       np_ims: np.ndarray, vis_masks: Optional[np.ndarray]):
        if self.cfg.use_gt_occlusion and gt_occlusion is not None:
            occl_info = [f["level"] for f in gt_occlusion]
            amount_of_occlusion = [f["amount"] for f in gt_occlusion]
        else:
            occl_info, amount_of_occlusion = predict_occlusion(np_ims, vis_masks, monodepth_results)
        occlusion_info = map_occlusion_levels(occl_info, amount_of_occlusion)
        # The first (query) frame is always occlusion free
        occlusion_info[0].level = OcclusionLevel.NO_OCCLUSION
        return occlusion_info

    def _get_pipeline_vis_masks(self, all_ims_pil: list[Image.Image], gt_vis_masks: Optional[np.ndarray],
                                query_mask: np.ndarray):
        if self.cfg.use_gt_vis_mask and gt_vis_masks is not None:
            vis_masks = gt_vis_masks
        else:
            vis_masks = predict_visible_masks(self.cfg.sam_checkpoint, all_ims_pil, query_mask)
        return vis_masks

    def init_trained_models(self, images: list[Image.Image], masks: np.ndarray, occlusion_info: list[OcclusionInfo]):
        self.video_diffusion.train_components(images, masks, occlusion_info)

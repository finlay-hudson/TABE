import torch

from src.eval_tabe51 import get_iou_results
from src.tabe.configs.runtime_config import RuntimeConfig
from src.tabe.datasets.custom import CustomDataset
from src.tabe.datasets.tabe import TABEDataset
from src.tabe.pipelines.video_diffusion_pipeline import VideoDiffusionPipeline
from src.tabe.pipelines.video_mask_gen_pipeline import VideoMaskGenerationPipeline
from src.tabe.utils.general_utils import set_all_seeds
from src.tabe.utils.mask_utils import convert_masks_to_correct_format
from src.tabe.utils.occlusion_utils import map_occlusion_levels
from src.tabe.utils.run_utils import setup
from src.tabe.utils.vis_utils import Visualiser


def _run(cfg: RuntimeConfig, ds_cls: TABEDataset | CustomDataset, vid_names) -> None:
    set_all_seeds(42)
    device = torch.device("cuda")

    for vid_name in vid_names:
        print(f"Running: {vid_name}")
        query_frame = 0
        all_ims_pil, gt_amodal_masks, gt_vis_masks, anno = ds_cls.get_data_for_vid(vid_name)
        all_ims_pil = all_ims_pil[query_frame:]
        if gt_amodal_masks is not None:
            gt_amodal_masks = gt_amodal_masks[query_frame:]
            gt_amodal_masks = convert_masks_to_correct_format(gt_amodal_masks)
        gt_vis_masks = gt_vis_masks[query_frame:]
        gt_vis_masks = convert_masks_to_correct_format(gt_vis_masks)
        gt_occlusion = anno["occlusion"][query_frame:] if anno is not None else None
        query_mask = gt_vis_masks[0]
        assert (query_mask > 0).any(), "Empty query masks"

        vis_out_dir = cfg.data.output_root / cfg.exp_name / ds_cls.name / "vis" / vid_name
        vis_out_dir.parent.mkdir(exist_ok=True, parents=True)

        trained_model_dir = vis_out_dir.parent.parent / "trained_models" / vid_name
        video_diffusion_pipeline = VideoDiffusionPipeline(cfg.video_diffusion, device, trained_model_dir)
        generation_pipeline = VideoMaskGenerationPipeline(cfg, video_diffusion_pipeline)

        outputs = generation_pipeline(all_ims_pil, query_mask, gt_vis_masks, gt_occlusion, gt_amodal_masks)

        visualiser = Visualiser(vis_out_dir)
        visualiser.visualise(outputs)

        # If we have ground truth information we can run IoU evaluation
        if gt_amodal_masks is not None and gt_occlusion is not None and gt_vis_masks is not None and cfg.img_padding == 0:
            vid_iou_results = get_iou_results(outputs.masks, gt_amodal_masks, gt_vis_masks,
                                              map_occlusion_levels([occl["level"] for occl in gt_occlusion]))
            print(vid_iou_results)


if __name__ == "__main__":
    runtime_cfg, dataset_cls, video_names = setup()
    _run(runtime_cfg, dataset_cls, video_names)

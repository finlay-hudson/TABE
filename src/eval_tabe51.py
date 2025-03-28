from natsort import natsorted
from pprint import pprint
from tqdm import tqdm

import numpy as np
from PIL import Image

from src.tabe.configs.runtime_config import RuntimeConfig
from src.tabe.datasets.tabe import TABEDataset
from src.tabe.utils.metric_utils import batch_iou, get_metrics_per_method, get_non_vis_iou
from src.tabe.utils.occlusion_utils import OcclusionLevel, map_occlusion_levels, OcclusionInfo
from src.tabe.utils.run_utils import setup


def get_mean_std_over_runs(results_per_run):
    # Metrics to compute
    metrics = ['mIOU', 'mIOUffo', 'mIOUfo', 'mIOUocc']

    # Compute statistics for each metric
    results = {}
    for metric in metrics:
        values = [results_per_run[gen][metric] for gen in results_per_run.keys()]

        mean_value = np.mean(values)
        std_value = np.std(values, ddof=1)  # Sample standard deviation
        sem_value = std_value / np.sqrt(len(values))  # Standard error of the mean

        results[metric] = {'mean': mean_value, 'std': std_value, 'sem': sem_value}

    return results


def get_iou_results(exp_pred_masks: np.ndarray, gt_amodal_masks: np.ndarray, gt_vis_masks: np.ndarray,
                    gt_occlusion_info: list[OcclusionInfo]):
    num_generations = exp_pred_masks.shape[0]
    all_ious = batch_iou(gt_amodal_masks[None].repeat(num_generations, 0), exp_pred_masks)
    all_non_vis_iou = get_non_vis_iou(exp_pred_masks, gt_amodal_masks[None].repeat(num_generations, 0),
                                      gt_vis_masks[None].repeat(num_generations, 0))

    # is fully occluded frames or item left scene
    fully_occluded_frames = gt_vis_masks.reshape(gt_vis_masks.shape[0], -1).sum(-1) < 1
    any_occluded_frames = np.array([o.level != OcclusionLevel.NO_OCCLUSION for o in gt_occlusion_info])
    non_vis_mask_gt = gt_amodal_masks - gt_vis_masks
    frames_with_non_vis_frames = non_vis_mask_gt.reshape(non_vis_mask_gt.shape[0], -1).sum(-1) > 0
    iou_res = {}
    for gen in range(num_generations):
        ious = all_ious[gen]
        non_vis_ious = all_non_vis_iou[gen]
        iou_res[f"gen: {gen}"] = {
            "mIOU": ious.mean(),
            "mIOUffo": ious[fully_occluded_frames].mean(),
            "mIOUfo": ious[any_occluded_frames].mean(),
            "mIOUocc": non_vis_ious[frames_with_non_vis_frames].mean()
        }

    return iou_res


def main(runtime_config: RuntimeConfig, vid_names: tuple):
    exp_dir = runtime_config.data.output_root / runtime_config.exp_name / "TABE51/vis"
    assert exp_dir.exists(), f"No experiment directory at {exp_dir}"

    print(f"Running eval for experiment at {exp_dir}")

    iou_results = {}
    for vid_name in tqdm(vid_names):
        print("Running eval for vid", vid_name)
        if not (exp_dir / vid_name).exists():
            raise ValueError(f"No outputs for video: {vid_name}")
        iou_results[vid_name] = {}
        all_ims_pil, gt_amodal_masks, gt_vis_masks, anno = TABEDataset(runtime_config.data.tabe).get_data_for_vid(
            vid_name)
        vid_exp_dir = exp_dir / vid_name
        num_experiments = len(list((vid_exp_dir / "final_outputs").glob("*")))
        exp_pred_masks = np.ones((num_experiments, *gt_amodal_masks.shape))
        for exp_num in range(num_experiments):
            pred_mask_fns = natsorted(list((vid_exp_dir / "final_outputs" / str(exp_num) / "masks").glob("*.png")))
            exp_pred_masks[exp_num] = np.stack([np.array(Image.open(fn)) for fn in pred_mask_fns])

        iou_results[vid_name] = get_iou_results(exp_pred_masks, gt_amodal_masks, gt_vis_masks,
                                                map_occlusion_levels([occl["level"] for occl in anno["occlusion"]]))
        pprint(iou_results[vid_name])

    mean_values_across_videos = get_metrics_per_method(iou_results)

    # Output the mean values across all videos for each method and metric
    print("\n")
    pprint(mean_values_across_videos)
    print("\n")

    mean_std_res = get_mean_std_over_runs(mean_values_across_videos)
    # Print results in mean ± error format
    for metric, stats in mean_std_res.items():
        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} (Std Dev)")
        print(f"{metric}: {stats['mean']:.4f} ± {stats['sem']:.4f} (SEM)\n")


if __name__ == "__main__":
    cfg, _, video_names = setup(just_eval=True)
    main(cfg, video_names)

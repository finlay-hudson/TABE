from pathlib import Path
from typing import Optional


from src.tabe.configs.runtime_config import DataConfigTABE
from src.tabe.datasets.utils import DatasetTypes
from src.tabe.datasets.base import Dataset

VID_NAMES = ['ball_bounce', 'plant_walk_behind_1', 'ball_throw_1', 'ducks_1', 'bottle_2', 'inside_table_1',
             'walk_outside_conversation_2', 'ball_office_4', 'ball_throw_slo_mo_2', 'outside_crosswalk_1', 'cat_car_3',
             'cars_1', 'ball_throw_slo_mo_1', 'bike_1', 'blue_towel_roll_1', 'cat_car_1', 'goose_1', 'goose_1-1',
             'door_close_1', 'ball_office_3', 'people_crossing', 'inside_table_3', 'goose_2', 'outside_table_moving_1',
             'stretch', 'inside_table_2', 'air_hockey_2', 'bottle_cap_roll_3', 'outside_table_stationary_1',
             'cat_car_4', 'door_close_2', 'ball_office_2', 'door_walkthrough_1', 'ball_roll', 'cat_car_2',
             'paper_plane_1', 'office_walk_infront', 'inside_table_upright_1', 'plant_stool_1', 'bottle_cap_roll_1',
             'goose_bike_distant_walk', 'cat_car_5', 'sliding_stools_1', 'ball_office_1', 'paper_plane_2',
             'plant_walk_behind_2', 'bottle_cap_roll_2', 'air_hockey_1', 'bus_1', 'fan_turn_1',
             'walk_outside_conversation_1']


def get_vid_names():
    return VID_NAMES


class TABEDataset(Dataset):
    def __init__(self, data_cfg: DataConfigTABE, ds_root: Optional[Path] = None):
        super().__init__(data_cfg, ds_root)
        self.name = DatasetTypes.TABE51.name


if __name__ == "__main__":
    ds = TABEDataset(DataConfigTABE())
    vid_names = get_vid_names()
    vid_name = vid_names[0]
    all_ims_pil, gt_masks, vis_masks, anno = ds.get_data_for_vid(vid_name)

from src.tabe.configs.runtime_config import DataConfig
from src.tabe.datasets.tabe import TABEDataset
from src.tabe.datasets.custom import CustomDataset
from src.tabe.datasets.tao_amodal import TAOAmodalDataset
from src.tabe.datasets.utils import DatasetTypes


def get_gt_data_cls(ds_type: DatasetTypes, cfg: DataConfig):
    if ds_type not in DatasetTypes:
        raise ValueError(f'Dataset type {ds_type} not supported')

    if ds_type == DatasetTypes.TABE51:
        return TABEDataset(cfg.tabe)
    elif ds_type == DatasetTypes.CUSTOM:
        return CustomDataset(cfg.custom)
    elif ds_type == DatasetTypes.TAOAMODAL:
        return TAOAmodalDataset(cfg.tao_amodal)
    else:
        raise NotImplementedError(f'Dataset type {ds_type} not supported')

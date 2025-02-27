from torch.utils.data import DataLoader
from src.configs import DataConfig, AudioFeatureConfig, EDAFeatureConfig
from src.data.datasets.hku956 import HKU956Dataset
from src.data.datasets.pmemo2019 import PMEmo2019Dataset

class DataLoaderBuilder:
    @staticmethod
    def build(data_config: DataConfig, audio_feature_config: AudioFeatureConfig,
              eda_feature_config: EDAFeatureConfig, split: str) -> DataLoader:
        # Instantiate datasets based on data_config and split
        # Implement prefetching logic with prefetch_size
        # Use a custom collate_fn to pad sequences on the left
        pass

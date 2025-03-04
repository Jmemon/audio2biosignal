from torch.utils.data import DataLoader
import torch
from src.configs import DataConfig, AudioEDAFeatureConfig, HKU956Config, PMEmo2019Config
from src.data.datasets.hku956 import HKU956Dataset, collate_fn as hku_collate_fn
from src.data.datasets.pmemo2019 import PMEmo2019Dataset, collate_fn as pmemo_collate_fn

class DataLoaderBuilder:
    @staticmethod
    def build(data_config: DataConfig, feature_config: AudioEDAFeatureConfig, split: str):
        dataset_mapping = {
            'hku956': (HKU956Dataset, hku_collate_fn, HKU956Config()),
            'pmemo2019': (PMEmo2019Dataset, pmemo_collate_fn, PMEmo2019Config()),
        }

        def create_dataloaders(dataset_names):
            dataloaders = []
            for name in dataset_names:
                DatasetClass, collate_fn, dataset_config = dataset_mapping[name.value]
                dataset = DatasetClass(dataset_config, feature_config)
                # Split dataset
                total_length = len(dataset)
                indices = list(range(total_length))
                split1 = int(total_length * 0.8)
                split2 = int(total_length * 0.9)
                if split == 'train':
                    selected_indices = indices[:split1]
                elif split == 'val':
                    selected_indices = indices[split1:split2]
                else:
                    selected_indices = indices[split2:]
                subset = torch.utils.data.Subset(dataset, selected_indices)
                dataloader = DataLoader(
                    subset,
                    batch_size=32,  # Adjust as needed
                    num_workers=data_config.num_workers,
                    prefetch_factor=data_config.prefetch_size,
                    collate_fn=collate_fn
                )
                dataloaders.append(dataloader)
            return dataloaders

        if split == 'train':
            return create_dataloaders(data_config.train_datasets)
        elif split == 'val':
            return create_dataloaders(data_config.val_datasets)
        else:
            return create_dataloaders(data_config.test_datasets)

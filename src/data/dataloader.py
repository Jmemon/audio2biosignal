from torch.utils.data import DataLoader
import torch
import multiprocessing
from typing import List
from src.configs import DataConfig, AudioEDAFeatureConfig, HKU956Config, PMEmo2019Config
from src.data.datasets.hku956 import HKU956Dataset, collate_fn as hku_collate_fn
from src.data.datasets.pmemo2019 import PMEmo2019Dataset, collate_fn as pmemo_collate_fn

# Use 'spawn' method for multiprocessing which is more compatible with boto3
# This is especially important on macOS
if multiprocessing.get_start_method() != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # If already set and we can't force it, we'll handle this in the DataLoader config
        pass

class DataLoaderBuilder:
    @staticmethod
    def build(data_config: DataConfig, feature_config: AudioEDAFeatureConfig, split: str) -> List[DataLoader]:
        """
        Constructs PyTorch DataLoaders for specified datasets with appropriate train/val/test splits.
        
        This factory method instantiates dataset objects, applies consistent dataset splitting,
        and configures DataLoaders with platform-aware multiprocessing settings to ensure
        compatibility across operating systems.
        
        Architecture:
            - Implements a factory pattern with O(n) complexity where n is the number of datasets
            - Uses fixed 80/10/10 train/val/test split ratios for all datasets
            - Handles multiprocessing context configuration with fallback to single-process loading
            - Maintains dataset-specific collate functions for proper batch construction
        
        Parameters:
            data_config (DataConfig): Configuration containing:
                - train_datasets/val_datasets/test_datasets: Lists of DatasetType enums
                - num_workers: Number of worker processes (0 disables multiprocessing)
                - prefetch_size: Number of batches to prefetch in background
            
            feature_config (AudioEDAFeatureConfig): Configuration for audio/EDA preprocessing
                - Controls sample rates, window sizes, and feature extraction parameters
                - Passed directly to dataset constructors
            
            split (str): Dataset split to return, must be one of:
                - 'train': Training dataset subset (first 80% of data)
                - 'val': Validation dataset subset (next 10% of data)
                - 'test': Test dataset subset (final 10% of data)
        
        Returns:
            List[DataLoader]: List of configured PyTorch DataLoaders, one per requested dataset
                - Empty list if no datasets are specified for the requested split
                - Each DataLoader uses fixed batch size of 32
        
        Behavior:
            - Automatically detects and handles multiprocessing compatibility issues
            - Falls back to single-process loading if 'spawn' method cannot be set
            - Uses dataset-specific collate functions for proper batch construction
            - Maintains fixed batch size of 32 regardless of dataset
        
        Integration:
            - Used in training pipelines to construct appropriate DataLoaders:
              ```
              train_loaders = DataLoaderBuilder.build(config.data, feature_config, 'train')
              ```
            - Requires dataset classes to implement __len__ for splitting calculations
            - Expects dataset_mapping to contain all datasets referenced in data_config
        
        Limitations:
            - Fixed 80/10/10 split ratios not configurable per dataset
            - Hard-coded batch size of 32 not configurable through parameters
            - Limited to two dataset types (HKU956, PMEmo2019)
            - No support for custom dataset splitting strategies (e.g., stratified)
        """
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
                
                # Determine if we should use multiprocessing
                # On macOS, use single process if spawn method couldn't be set
                use_workers = data_config.num_workers
                if multiprocessing.get_start_method() != 'spawn' and use_workers > 0:
                    print(f"Warning: Multiprocessing start method is not 'spawn'. Using single process data loading.")
                    use_workers = 0
                
                dataloader = DataLoader(
                    subset,
                    batch_size=32,  # Adjust as needed
                    num_workers=use_workers,
                    prefetch_factor=data_config.prefetch_size if use_workers > 0 else 2,
                    collate_fn=collate_fn,
                    multiprocessing_context='spawn' if use_workers > 0 else None
                )
                dataloaders.append(dataloader)
            return dataloaders

        if split == 'train':
            return create_dataloaders(data_config.train_datasets)
        elif split == 'val':
            return create_dataloaders(data_config.val_datasets)
        else:
            return create_dataloaders(data_config.test_datasets)

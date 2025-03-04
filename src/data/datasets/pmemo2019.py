import torch
from torch.utils.data import Dataset
from src.configs import PMEmo2019Config, AudioEDAFeatureConfig
from src.data.audio_preprocessing import preprocess_audio
from src.data.eda_preprocessing import preprocess_eda
from src.utilities import S3FileManager
import csv
import pandas as pd

def collate_fn(batch):
    audio_tensors, eda_tensors = zip(*batch)
    max_length = max(tensor.size(1) for tensor in audio_tensors)
    audio_features = audio_tensors[0].size(0)

    padded_audio = torch.zeros(len(batch), audio_features, max_length)
    padded_eda = torch.zeros(len(batch), max_length)

    for i, (audio, eda) in enumerate(batch):
        audio_length = audio.size(1)
        eda_length = eda.size(1)
        padded_audio[i, :, -audio_length:] = audio
        padded_eda[i, -eda_length:] = eda

    return padded_audio, padded_eda

class PMEmo2019Dataset(Dataset):
    def __init__(self, dataset_config: PMEmo2019Config, feature_config: AudioEDAFeatureConfig):
        self.dataset_config = dataset_config
        self.feature_config = feature_config
        self.s3_manager = S3FileManager()
        self._audio_files = {}
        self._eda_files = {}
        self.examples = []

        # Load audio metadata
        metadata_csv_path = "s3://audio2biosignal-train-data/PMEmo2019/metadata.csv"
        local_metadata_csv = self.s3_manager.download_file(metadata_csv_path)
        metadata_df = pd.read_csv(local_metadata_csv)
        for _, row in metadata_df.iterrows():
            music_id = str(row['musicId'])
            file_name = row['fileName']
            audio_s3_path = f"s3://audio2biosignal-train-data/PMEmo2019/chorus/{file_name}"
            self._audio_files[music_id] = audio_s3_path

        # Load EDA files
        # List all EDA files in the S3 bucket
        # Assuming implementation using boto3
        # For illustration, let's assume self._eda_files is populated

        # Create examples
        for (music_id, subject_id), eda_s3_path in self._eda_files.items():
            audio_s3_path = self._audio_files.get(music_id)
            if audio_s3_path:
                self.examples.append({
                    (subject_id, music_id): (audio_s3_path, eda_s3_path)
                })

    def _load_audio_file(self, audio_file_s3_path: str) -> torch.Tensor:
        local_audio_path = self.s3_manager.download_file(audio_file_s3_path)
        audio_tensor = preprocess_audio(local_audio_path, self.feature_config)
        return audio_tensor

    def _load_eda_file(self, eda_file_s3_path: str, subject_id: str) -> torch.Tensor:
        local_eda_path = self.s3_manager.download_file(eda_file_s3_path)
        eda_df = pd.read_csv(local_eda_path)
        time_col = eda_df.columns[0]
        eda_series = eda_df[subject_id]
        eda_tensor = preprocess_eda(eda_series.values, self.feature_config)
        return eda_tensor

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        (subject_id, music_id), (audio_s3_path, eda_s3_path) = list(example.items())[0]
        audio_tensor = self._load_audio_file(audio_s3_path)
        eda_tensor = self._load_eda_file(eda_s3_path, subject_id)
        # Prefetch next examples
        if index + 1 < len(self.examples):
            next_examples = self.examples[index+1:index+5]
            s3_paths = []
            for ex in next_examples:
                _, (next_audio_s3_path, next_eda_s3_path) = list(ex.items())[0]
                s3_paths.extend([next_audio_s3_path, next_eda_s3_path])
            self.s3_manager.prefetch_files(s3_paths)
        return audio_tensor, eda_tensor

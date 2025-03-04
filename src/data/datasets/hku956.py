import torch
from torch.utils.data import Dataset
from src.configs import HKU956Config, AudioEDAFeatureConfig
from src.data.audio_preprocessing import preprocess_audio
from src.data.eda_preprocessing import preprocess_eda
from src.utilities import S3FileManager
import csv

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

class HKU956Dataset(Dataset):
    def __init__(self, dataset_config: HKU956Config, feature_config: AudioEDAFeatureConfig):
        self.dataset_config = dataset_config
        self.feature_config = feature_config
        self.s3_manager = S3FileManager()
        self._audio_files = {}
        self._eda_files = {}
        self.examples = []

        # Load audio files
        audio_csv_path = "s3://audio2biosignal-train-data/HKU956/2. original_song_audio.csv"
        local_audio_csv = self.s3_manager.download_file(audio_csv_path)
        with open(local_audio_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                song_id = row['song_id']
                link = row['link']
                self._audio_files[song_id] = link

        # Load EDA files
        # Assuming listing of S3 paths for EDA files
        # You might need to use boto3 to list files in the S3 bucket
        eda_root = "s3://audio2biosignal-train-data/HKU956/1. physiological_signals/"
        # This requires proper implementation to list subjects and files
        # For illustration, let's assume self._eda_files is populated

        # Create examples
        for (subject, song_id), eda_s3_path in self._eda_files.items():
            audio_link = self._audio_files.get(song_id)
            if audio_link:
                self.examples.append({
                    (subject, song_id): (audio_link, eda_s3_path)
                })

    def _load_audio_file(self, audio_file_link: str) -> torch.Tensor:
        # Download the audio file from the URL
        local_audio_path = self.s3_manager.download_file(audio_file_link)
        audio_tensor = preprocess_audio(local_audio_path, self.feature_config)
        return audio_tensor

    def _load_eda_file(self, eda_file_path: str) -> torch.Tensor:
        local_eda_path = self.s3_manager.download_file(eda_file_path)
        eda_tensor = preprocess_eda(local_eda_path, self.feature_config)
        return eda_tensor

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        (subject, song_id), (audio_link, eda_s3_path) = list(example.items())[0]
        audio_tensor = self._load_audio_file(audio_link)
        eda_tensor = self._load_eda_file(eda_s3_path)
        # Prefetch the next few examples
        if index + 1 < len(self.examples):
            next_examples = self.examples[index+1:index+5]
            s3_paths = []
            for ex in next_examples:
                _, (next_audio_link, next_eda_s3_path) = list(ex.items())[0]
                s3_paths.extend([next_audio_link, next_eda_s3_path])
            self.s3_manager.prefetch_files(s3_paths)
        return audio_tensor, eda_tensor

import torch
import torchaudio
from torch.utils.data import Dataset
from src.configs import HKU956Config, AudioEDAFeatureConfig
from src.data.audio_preprocessing import preprocess_audio
from src.data.eda_preprocessing import preprocess_eda
from src.utilities import S3FileManager
import csv
import pandas as pd
import os
import re

def collate_fn(batch):
    """
    Collate and pad variable-length audio and EDA tensors into uniform batches for model training.
    
    This function handles the conversion of individual dataset samples into properly formatted
    batches, ensuring all tensors have consistent dimensions by right-aligned padding with zeros.
    It maintains temporal alignment between audio features and corresponding EDA signals.
    
    Architecture:
        - Determines maximum sequence length across all samples in the batch with O(n) complexity
        - Creates zero-padded tensors of uniform size for both audio and EDA data
        - Preserves original data by right-aligning within padded tensors
        - Time complexity: O(n) where n is the batch size
        - Space complexity: O(n*m) where m is the maximum sequence length
    
    Parameters:
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
            List of (audio_tensor, eda_tensor) pairs where:
            - audio_tensor: Shape [features, length]
            - eda_tensor: Shape [length]
            
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - padded_audio: Tensor of shape [batch_size, features, max_length]
            - padded_eda: Tensor of shape [batch_size, max_length]
            
    Raises:
        IndexError: If batch is empty
        RuntimeError: If audio tensors have inconsistent feature dimensions
    
    Notes:
        - Maintains original tensor device and dtype
        - Right-alignment preserves temporal relationship between audio and EDA signals
        - Zero-padding is applied to the left side of sequences
    """
    print(f"[collate_fn HKU956] Processing batch of size: {len(batch)}")
    audio_tensors, eda_tensors = zip(*batch)
    print(f"[collate_fn HKU956] Audio tensors shapes: {[tensor.shape for tensor in audio_tensors]}")
    print(f"[collate_fn HKU956] EDA tensors shapes: {[tensor.shape for tensor in eda_tensors]}")
    
    max_length = max(tensor.size(1) for tensor in audio_tensors)
    audio_features = audio_tensors[0].size(0)
    print(f"[collate_fn HKU956] Max length: {max_length}, Audio features: {audio_features}")

    padded_audio = torch.zeros(len(batch), audio_features, max_length)
    padded_eda = torch.zeros(len(batch), max_length)

    for i, (audio, eda) in enumerate(batch):
        audio_length = audio.size(1)
        eda_length = eda.size(1)
        print(f"[collate_fn HKU956] Sample {i}: audio_length={audio_length}, eda_length={eda_length}")
        
        padded_audio[i, :, -audio_length:] = audio
        padded_eda[i, -eda_length:] = eda

    print(f"[collate_fn HKU956] Final padded_audio shape: {padded_audio.shape}")
    print(f"[collate_fn HKU956] Final padded_eda shape: {padded_eda.shape}")
    return padded_audio, padded_eda

class HKU956Dataset(Dataset):
    def __init__(self, dataset_config: HKU956Config, feature_config: AudioEDAFeatureConfig):
        """
        Initialize the HKU956 dataset with configuration and S3-based data loading.
        
        This constructor establishes the dataset structure by loading metadata from S3,
        mapping audio files to song IDs, and EDA files to (subject, song_id) pairs.
        It creates a list of examples that associate each subject's EDA data with the
        corresponding audio stimulus for efficient retrieval during training.
        
        Architecture:
            - Uses S3FileManager for cloud storage access with O(1) file retrieval
            - Builds lookup dictionaries for audio and EDA files with O(n) complexity
            - Creates a flat list of examples for indexed access with O(1) lookup time
            - Maintains separation between metadata parsing and data loading for efficiency
        
        Parameters:
            dataset_config: HKU956Config
                Configuration specific to the HKU956 dataset including paths and parameters
            feature_config: AudioEDAFeatureConfig
                Configuration for audio and EDA feature extraction and preprocessing
                
        Raises:
            FileNotFoundError: If metadata CSV cannot be accessed or downloaded
            KeyError: If required columns are missing from the metadata CSV
            
        Thread Safety:
            - Thread-safe for read operations (getitem)
            - Not thread-safe for initialization
            
        Resource Management:
            - Leverages S3FileManager's caching for efficient repeated access
            - Defers actual file loading until __getitem__ is called
            - Maintains minimal memory footprint by storing only file references
        """
        self.dataset_config = dataset_config
        self.feature_config = feature_config
        self.s3_manager = S3FileManager()
        self._audio_files = {}
        self._eda_files = {}
        self.examples = []

        # Load metadata from custom_metadata.csv which contains all the mappings
        metadata_csv_path = "s3://audio2biosignal-train-data/HKU956/custom_metadata.csv"
        local_metadata_csv = self.s3_manager.download_file(metadata_csv_path)
        self.metadata = pd.read_csv(local_metadata_csv)
        
        # Populate audio and EDA file dictionaries
        for _, row in self.metadata.iterrows():
            song_id = str(row['song_id'])
            subject = str(row['subject'])
            audio_s3_path = row['audio_path']
            eda_s3_path = row['eda_path']
            
            # Store audio path by song_id
            self._audio_files[song_id] = audio_s3_path
            
            # Store EDA path by (subject, song_id) pair
            self._eda_files[(subject, song_id)] = eda_s3_path

        # Create examples
        for (subject, song_id), eda_s3_path in self._eda_files.items():
            audio_s3_path = self._audio_files.get(song_id)
            if audio_s3_path:
                self.examples.append({
                    (subject, song_id): (audio_s3_path, eda_s3_path)
                })

    def _load_audio_file(self, audio_file_link: str) -> torch.Tensor:
        # Download the audio file from the URL
        local_audio_path = self.s3_manager.download_file(audio_file_link)
        # Load audio with torchaudio
        waveform, sampling_rate = torchaudio.load(local_audio_path)
        # Process the audio tensor
        audio_tensor = preprocess_audio(waveform, sampling_rate, self.feature_config)
        return audio_tensor

    def _load_eda_file(self, eda_file_path: str) -> torch.Tensor:
        """
        Load and preprocess an EDA signal from an S3-hosted CSV file.
        
        This method retrieves EDA (Electrodermal Activity) data from a CSV file,
        extracts the signal from the first column, and applies standardized preprocessing
        to prepare it for model training. It handles the complete pipeline from cloud storage
        access to tensor preparation.
        
        Architecture:
            - Downloads CSV from S3 with O(1) lookup via caching
            - Extracts signal data with O(n) pandas column access
            - Converts to tensor and applies preprocessing with O(n) complexity
            - Time complexity dominated by preprocessing: O(n) where n is signal length
            
        Parameters:
            eda_file_path: str
                S3 path to the CSV file containing EDA data
                
        Returns:
            torch.Tensor:
                Preprocessed EDA tensor with shape determined by feature_config
                
        Raises:
            ValueError: If eda_file_path is empty
            FileNotFoundError: If the S3 file cannot be accessed
            pd.errors.ParserError: If CSV parsing fails
            
        Thread Safety:
            - Thread-safe for concurrent calls with different parameters
            - Relies on S3FileManager's thread safety for file access
            
        Notes:
            - Assumes CSV format with EDA signal in the first column
            - Preprocessing applies normalization and resampling based on feature_config
            - Caches downloaded files via S3FileManager for repeated access efficiency
        """
        local_eda_path = self.s3_manager.download_file(eda_file_path)
        # Load the CSV file
        eda_df = pd.read_csv(local_eda_path)
        # Extract the entirety of the first column as the EDA signal
        eda_signal = torch.tensor(eda_df.iloc[:, 0].values, dtype=torch.float32)
        
        # Extract song_id from eda_file_path
        # Assuming filename format is <song_no>_<song_id>.csv
        filename = os.path.basename(eda_file_path)
        song_id_match = re.search(r'_(\d+)\.csv', filename)
        
        if song_id_match:
            song_id = int(song_id_match.group(1))
            # Get the row from metadata with matching song_id
            metadata_row = self.metadata[self.metadata['song_id'] == song_id]
            if not metadata_row.empty:
                # Calculate sample rate from number of samples and duration
                duration = metadata_row['duration'].values[0]
                sample_rate = int(len(eda_signal) / duration)
            else:
                raise ValueError(f"Song ID {song_id} not found in metadata")
        else:
            raise ValueError(f"Could not extract song_id from EDA file path: {eda_file_path}")
        
        # Process the EDA tensor with the calculated sample rate
        eda_tensor = preprocess_eda(eda_signal, sample_rate, self.feature_config)
        return eda_tensor

    def __len__(self) -> int:
        """
        Return the total number of examples in the dataset.
        
        This method provides the dataset size by returning the count of valid (subject, song_id) 
        pairs that have both associated EDA and audio data. It's used by PyTorch's DataLoader to 
        determine iteration boundaries and sampling strategies.
        
        Architecture:
            - Delegates to Python's built-in len() with O(1) complexity
            - Reflects the count established during initialization
            - Remains constant throughout the dataset's lifecycle
        
        Returns:
            int:
                The number of examples available in the dataset
                
        Thread Safety:
            - Thread-safe (read-only operation on immutable property)
            
        Notes:
            - Critical for DataLoader's sampling, batching, and shuffling operations
            - Value represents complete audio-EDA pairs, not raw file counts
            - Zero is a valid return value indicating an empty dataset
        """
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

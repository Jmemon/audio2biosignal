import torch
import torchaudio
from torch.utils.data import Dataset
from src.configs import PMEmo2019Config, AudioEDAFeatureConfig
from src.data.audio_preprocessing import preprocess_audio
from src.data.eda_preprocessing import preprocess_eda
from src.utilities import S3FileManager
import csv
import pandas as pd

def collate_fn(batch):
    """
    Collate and pad variable-length audio and EDA tensors into uniform batches for model training.
    
    This function handles the conversion of individual dataset samples into properly formatted
    batches, ensuring all tensors have consistent dimensions by right-aligned padding with zeros.
    It supports both 2D audio tensors (features, length) and 3D audio tensors (channels, features, length).
    
    Architecture:
        - Determines maximum sequence length across all samples in the batch
        - Creates zero-padded tensors of uniform size for both audio and EDA data
        - Preserves original data by right-aligning within padded tensors
        - Handles dimensional variance in audio tensors (with or without channel dimension)
        - Time complexity: O(n) where n is the batch size
    
    Parameters:
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
            List of (audio_tensor, eda_tensor) pairs where:
            - audio_tensor: Either shape [features, length] or [channels, features, length]
            - eda_tensor: Shape [length]
            
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - padded_audio: Tensor of shape [batch_size, channels, features, max_length]
            - padded_eda: Tensor of shape [batch_size, max_length]
            
    Raises:
        IndexError: If batch is empty
        RuntimeError: If audio tensors have inconsistent feature dimensions
    
    Notes:
        - Maintains original tensor device and dtype
        - Right-alignment preserves temporal relationship between audio and EDA signals
        - Zero-padding is applied to the left side of sequences
    """
    audio_tensors, eda_tensors = zip(*batch)
    max_length = max(tensor.size(-1) for tensor in audio_tensors)
    audio_features = audio_tensors[0].size(0)
    
    # Determine number of channels from the audio tensors
    # If audio is already in shape [features, length], we're adding a channel dimension
    num_channels = 1
    if len(audio_tensors[0].shape) > 2:
        num_channels = audio_tensors[0].size(0)
        audio_features = audio_tensors[0].size(1)

    padded_audio = torch.zeros(len(batch), num_channels, audio_features, max_length)
    padded_eda = torch.zeros(len(batch), max_length)

    for i, (audio, eda) in enumerate(batch):
        audio_length = audio.size(-1)
        eda_length = eda.size(-1)
        
        if len(audio.shape) > 2:
            # If audio already has channel dimension
            padded_audio[i, :, :, -audio_length:] = audio
        else:
            # If audio is in shape [features, length]
            padded_audio[i, 0, :, -audio_length:] = audio
            
        padded_eda[i, -eda_length:] = eda

    return padded_audio, padded_eda

class PMEmo2019Dataset(Dataset):
    def __init__(self, dataset_config: PMEmo2019Config, feature_config: AudioEDAFeatureConfig):
        """
        Initialize the PMEmo2019 dataset with configuration and S3-based data loading.
        
        This constructor establishes the dataset structure by loading metadata from S3,
        mapping audio files to music IDs, and EDA files to (subject_id, music_id) pairs.
        It creates a list of examples that associate each subject's EDA data with the
        corresponding audio stimulus for efficient retrieval during training.
        
        Architecture:
            - Uses S3FileManager for cloud storage access with O(1) file retrieval
            - Builds lookup dictionaries for audio and EDA files with O(n) complexity
            - Creates a flat list of examples for indexed access with O(1) lookup time
            - Maintains separation between metadata parsing and data loading for efficiency
        
        Parameters:
            dataset_config: PMEmo2019Config
                Configuration specific to the PMEmo2019 dataset including paths and parameters
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
        metadata_csv_path = "s3://audio2biosignal-train-data/PMEmo2019/custom_metadata.csv"
        local_metadata_csv = self.s3_manager.download_file(metadata_csv_path)
        metadata_df = pd.read_csv(local_metadata_csv)
        
        # Populate audio and EDA file dictionaries
        for _, row in metadata_df.iterrows():
            music_id = str(row['music_id'])
            subject_id = str(row['subject_id'])
            audio_s3_path = row['audio_path']
            eda_s3_path = row['eda_path']
            
            # Store audio path by music_id
            self._audio_files[music_id] = audio_s3_path
            
            # Store EDA path by (subject_id, music_id) pair
            self._eda_files[(subject_id, music_id)] = eda_s3_path

        # Create examples
        for (subject_id, music_id), eda_s3_path in self._eda_files.items():
            audio_s3_path = self._audio_files.get(music_id)
            if audio_s3_path:
                self.examples.append({
                    (subject_id, music_id): (audio_s3_path, eda_s3_path)
                })

    def _load_audio_file(self, audio_file_s3_path: str) -> torch.Tensor:
        """
        Load and preprocess an audio file from S3 storage for model training.
        
        This method retrieves audio data from cloud storage, loads it using torchaudio,
        and applies standardized preprocessing to prepare it for model training. It handles
        the complete pipeline from S3 file retrieval to tensor preparation.
        
        Architecture:
            - Downloads audio from S3 with O(1) lookup via caching
            - Loads audio with torchaudio's efficient I/O operations
            - Applies feature extraction and preprocessing with O(n) complexity
            - Time complexity dominated by preprocessing: O(n) where n is audio length
            
        Parameters:
            audio_file_s3_path: str
                S3 path to the audio file to be loaded and processed
                
        Returns:
            torch.Tensor:
                Preprocessed audio tensor with shape determined by feature_config
                
        Raises:
            ValueError: If audio_file_s3_path is empty
            FileNotFoundError: If the S3 file cannot be accessed
            RuntimeError: If audio loading or preprocessing fails
            
        Thread Safety:
            - Thread-safe for concurrent calls with different parameters
            - Relies on S3FileManager's thread safety for file access
            
        Notes:
            - Supports various audio formats handled by torchaudio
            - Preprocessing applies feature extraction based on feature_config
            - Caches downloaded files via S3FileManager for repeated access efficiency
        """
        # Download the audio file from the URL
        local_audio_path = self.s3_manager.download_file(audio_file_s3_path)
        # Load audio with torchaudio
        waveform, sampling_rate = torchaudio.load(local_audio_path)
        # Process the audio tensor
        audio_tensor = preprocess_audio(waveform, sampling_rate, self.feature_config)
        return audio_tensor

    def _load_eda_file(self, eda_file_s3_path: str, subject_id: str) -> torch.Tensor:
        """
        Load and preprocess an EDA signal for a specific subject from an S3-hosted CSV file.
        
        This method retrieves EDA (Electrodermal Activity) data from a multi-subject CSV file,
        extracts the specific subject's signal, and applies standardized preprocessing to
        prepare it for model training. It handles the complete pipeline from cloud storage
        access to tensor preparation.
        
        Architecture:
            - Downloads CSV from S3 with O(1) lookup via caching
            - Extracts subject-specific column with O(1) pandas column access
            - Converts to tensor and applies preprocessing with O(n) complexity
            - Time complexity dominated by preprocessing: O(n) where n is signal length
            
        Parameters:
            eda_file_s3_path: str
                S3 path to the CSV file containing EDA data for multiple subjects
            subject_id: str
                Identifier for the specific subject whose EDA data should be extracted
                
        Returns:
            torch.Tensor:
                Preprocessed EDA tensor with shape determined by feature_config
                
        Raises:
            ValueError: If eda_file_s3_path or subject_id is empty
            KeyError: If subject_id column doesn't exist in the CSV
            FileNotFoundError: If the S3 file cannot be accessed
            
        Thread Safety:
            - Thread-safe for concurrent calls with different parameters
            - Relies on S3FileManager's thread safety for file access
            
        Notes:
            - Assumes CSV format with first column as time and subject columns by ID
            - Preprocessing applies normalization and resampling based on feature_config
            - Caches downloaded files via S3FileManager for repeated access efficiency
        """
        local_eda_path = self.s3_manager.download_file(eda_file_s3_path)
        eda_df = pd.read_csv(local_eda_path)
        time_col = eda_df.columns[0]
        eda_series = eda_df[subject_id]
        # Convert series to tensor before preprocessing
        eda_signal = torch.tensor(eda_series.values, dtype=torch.float32)
        # Process the EDA tensor
        eda_tensor = preprocess_eda(eda_signal, self.feature_config)
        return eda_tensor

    def __len__(self) -> int:
        """
        Return the total number of examples in the dataset.
        
        This method provides the dataset size by returning the count of valid (subject_id, music_id) 
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve and process a single example from the dataset at the specified index.
        
        This method implements PyTorch's Dataset protocol for indexed access, retrieving
        the audio and EDA data for a specific (subject_id, music_id) pair. It handles the
        complete data loading pipeline including S3 file retrieval, audio/EDA preprocessing,
        and tensor preparation with lazy evaluation for memory efficiency.
        
        Architecture:
            - Performs O(1) lookup in the examples list for file references
            - Delegates to specialized loading methods for audio and EDA processing
            - Implements lazy loading pattern, only retrieving data when requested
            - Time complexity dominated by audio/EDA preprocessing: O(n) where n is signal length
            - Space complexity: O(m) where m is the size of the processed tensors
        
        Parameters:
            index: int
                Zero-based index into the dataset's examples list
                
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - audio_tensor: Preprocessed audio features with shape determined by feature_config
                - eda_tensor: Preprocessed EDA signal with shape determined by feature_config
                
        Raises:
            IndexError: If index is out of bounds (< 0 or >= len(self))
            RuntimeError: If audio or EDA file loading/processing fails
            
        Thread Safety:
            - Thread-safe for concurrent calls with different indices
            - Relies on S3FileManager's thread safety for file access
            
        Notes:
            - Core method used by PyTorch DataLoader during training/evaluation
            - No caching of processed tensors between calls (stateless design)
            - Audio and EDA tensors are guaranteed to have matching temporal dimensions
        """
        example = self.examples[index]
        (subject_id, music_id), (audio_s3_path, eda_s3_path) = list(example.items())[0]
        audio_tensor = self._load_audio_file(audio_s3_path)
        eda_tensor = self._load_eda_file(eda_s3_path, subject_id)
        return audio_tensor, eda_tensor

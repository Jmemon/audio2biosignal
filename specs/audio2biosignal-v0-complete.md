

## Implementation Notes
- Comment everything



## Low-Level Tasks
1. Move DatasetTypes enum.
UPDATE src/configs.py:
    MOVE DatasetTypes FROM src/data/datasets/types.py TO top of src/configs.py

DELETE src/data/datasets/types.py

2. Create utilities file with S3 file management subsystem.
CREATE src/utilities.py:
    CREATE S3FileManager class:
        - Implement as a singleton pattern
        - Support downloading files from S3 buckets
        - Implement prefetching mechanism to prepare files ahead of time
        - Use temporary storage to avoid persisting files
        - Include a cache with LRU (Least Recently Used) eviction policy
        - Support concurrent downloads with thread pool
        - Provide both blocking and non-blocking download methods
        - Include methods:
            - download_file(s3_path) -> Path: Blocking download
            - prefetch_files(s3_paths) -> None: Non-blocking batch prefetch
            - get_file(s3_path) -> Optional[Path]: Non-blocking cache check
            - clear_cache() -> None: Remove all cached files
        
    CREATE s3_download() -> Path:
        - Wrapper around S3FileManager.download_file for simple use cases
        
    IMPLEMENTATION DETAILS:
        - Use ThreadPoolExecutor for concurrent downloads
        - Use PriorityQueue for download scheduling
        - Store files in temporary directory that's cleaned up on exit
        - Track in-progress downloads to avoid duplicate requests
        - Support configurable cache size limits
        - Handle S3 path parsing (s3://bucket/key format)

3. Implement HKU956Dataset
   - Integrate with S3FileManager for efficient file loading
   - Implement prefetching in __getitem__ to prepare next batch
   - Use temporary files for audio and EDA data
   - Implement proper cleanup to avoid memory leaks

4. Implement PMEmoDataset

5. DataLoaderBuilder
   - Implement custom DataLoader with prefetching capabilities
   - Support configurable prefetch size
   - Integrate with S3FileManager for batch prefetching
   - Implement custom collate function for padding sequences

6. CNNDecoder

7. LSTMDecoder

8. TransformerDecoder

9. TransformerEncoder

10. WavenetEncoder

11. Instantiate scheduler

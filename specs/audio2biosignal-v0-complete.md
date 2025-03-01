

## Implementation Notes
- Comment everything



## Low-Level Tasks
1. Move DatasetTypes enum.
UPDATE src/configs.py:
    MOVE DatasetTypes FROM src/data/datasets/types.py TO top of src/configs.py

DELETE src/data/datasets/types.py

2. Create utilities file with s3 interaction functions.
CREATE src/utilities.py:
    CREATE s3_download() -> Path:
        with s3 downloading

3. Implement HKU956Dataset

4. Implement PMEmoDataset

5. DataLoaderBuilder

6. CNNDecoder

7. LSTMDecoder

8. TransformerDecoder

9. TransformerEncoder

10. WavenetEncoder

11. Instantiate scheduler
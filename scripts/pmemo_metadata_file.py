#!/usr/bin/env python3
"""
Script to create a custom metadata file for the PMEmo2019 dataset.
This script extracts music_id, subject_id, eda_path, and audio_path information
and saves it to a CSV file, which is then uploaded to S3.
"""

import boto3
import csv
import pandas as pd
import tempfile
from pathlib import Path
import os
import sys

# Add the project root to the Python path to ensure src module is accessible
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utilities import S3FileManager

def main():
    """
    Main function to create and upload the metadata file.
    """
    print("Creating PMEmo2019 custom metadata file...")
    
    # Initialize S3 file manager
    s3_manager = S3FileManager()
    s3_client = s3_manager._get_s3_client()
    
    # Dictionary to store audio file paths
    audio_files = {}
    
    # Dictionary to store EDA file paths
    eda_files = {}
    
    # Load audio metadata
    print("Loading audio metadata...")
    metadata_csv_path = "s3://audio2biosignal-train-data/PMEmo2019/metadata.csv"
    local_metadata_csv = s3_manager.download_file(metadata_csv_path)
    metadata_df = pd.read_csv(local_metadata_csv)
    
    for _, row in metadata_df.iterrows():
        music_id = str(row['musicId'])
        file_name = row['fileName']
        audio_s3_path = f"s3://audio2biosignal-train-data/PMEmo2019/chorus/{file_name}"
        audio_files[music_id] = audio_s3_path
    
    # Load EDA files
    print("Loading EDA files...")
    eda_base_path = "s3://audio2biosignal-train-data/PMEmo2019/EDA"
    bucket, prefix = s3_manager._parse_s3_path(eda_base_path)
    
    # List objects in the EDA directory
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    # Process each EDA file
    for obj in response.get('Contents', []):
        file_key = obj['Key']
        file_name = file_key.split('/')[-1]
        
        # Extract music_id from filename (assuming format: <music_id>_EDA.csv)
        if not file_name.endswith('_EDA.csv'):
            continue
            
        music_id = file_name.split('_')[0]
        eda_s3_path = f"s3://{bucket}/{file_key}"
        
        # Download the file to get subject IDs (column headers)
        local_eda_path = s3_manager.download_file(eda_s3_path)
        
        # Read the first row to get subject IDs
        with open(local_eda_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
        
        # First column is time, rest are subject IDs
        subject_ids = header[1:]
        
        # Map each (subject_id, music_id) pair to this EDA file
        for subject_id in subject_ids:
            eda_files[(subject_id, music_id)] = eda_s3_path
    
    # Create metadata rows
    print("Creating metadata rows...")
    metadata_rows = []
    
    for (subject_id, music_id), eda_path in eda_files.items():
        audio_path = audio_files.get(music_id)
        if audio_path:  # Only include if we have both audio and EDA data
            metadata_rows.append({
                'music_id': music_id,
                'subject_id': subject_id,
                'eda_path': eda_path,
                'audio_path': audio_path
            })
    
    # Create a temporary CSV file
    print(f"Writing {len(metadata_rows)} rows to CSV...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        fieldnames = ['music_id', 'subject_id', 'eda_path', 'audio_path']
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
        temp_file_path = temp_file.name
    
    # Upload the file to S3
    print("Uploading to S3...")
    bucket, _ = s3_manager._parse_s3_path("s3://audio2biosignal-train-data/PMEmo2019/")
    s3_client.upload_file(
        temp_file_path, 
        bucket, 
        "PMEmo2019/custom_metadata.csv"
    )
    
    # Clean up the temporary file
    os.unlink(temp_file_path)
    
    print("Custom metadata file created and uploaded successfully!")
    print(f"S3 Path: s3://audio2biosignal-train-data/PMEmo2019/custom_metadata.csv")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to create a custom metadata file for the HKU956 dataset.
This script extracts subject, song_id, audio_path, and eda_path information
and saves it to a CSV file, which is then uploaded to S3.
"""

import boto3
import csv
import pandas as pd
import tempfile
from pathlib import Path
import os
import sys
import torchaudio
import urllib.request
import concurrent.futures
from typing import Dict, List
import time

# Add the project root to the Python path to ensure src module is accessible
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utilities import S3FileManager

def main():
    """
    Main function to create and upload the metadata file.
    """
    print("Creating HKU956 custom metadata file...")
    
    # Initialize S3 file manager
    s3_manager = S3FileManager()
    s3_client = s3_manager._get_s3_client()
    
    # Dictionary to cache song durations to avoid redundant downloads
    song_duration_cache = {}
    
    # Load AV ratings to get participant_id, song_id, and song_no
    print("Loading AV ratings data...")
    av_ratings_path = "s3://audio2biosignal-train-data/HKU956/3. AV_ratings.csv"
    local_av_ratings_csv = s3_manager.download_file(av_ratings_path)
    av_ratings_df = pd.read_csv(local_av_ratings_csv)
    
    # Load audio metadata
    print("Loading audio metadata...")
    audio_metadata_path = "s3://audio2biosignal-train-data/HKU956/2. original_song_audio.csv"
    local_audio_metadata_csv = s3_manager.download_file(audio_metadata_path)
    audio_metadata_df = pd.read_csv(local_audio_metadata_csv)
    
    # Create a dictionary to map song_id to audio link
    song_id_to_audio = dict(zip(audio_metadata_df['song_id'], audio_metadata_df['link']))
    
    def process_participant_song(participant_id: str, song_id: str, song_no: str, 
                                audio_path: str, s3_manager: S3FileManager, song_duration_cache: Dict[str, float]) -> Dict:
        """
        Process a single participant-song combination.
        
        Args:
            participant_id: The participant ID
            song_id: The song ID
            song_no: The song number
            audio_path: Path to the audio file
            s3_manager: S3FileManager instance
            song_duration_cache: Dictionary mapping song_id to duration
            
        Returns:
            Dictionary with metadata or None if processing failed
        """
        # Construct EDA file path
        eda_file_name = f"{song_no}_{song_id}.csv"
        eda_s3_path = f"s3://audio2biosignal-train-data/HKU956/1. physiological_signals/{participant_id}/EDA/{eda_file_name}"
        
        try:
            # Check if EDA file exists
            bucket, key = s3_manager._parse_s3_path(eda_s3_path)
            s3_client = s3_manager._get_s3_client()
            s3_client.head_object(Bucket=bucket, Key=key)
            
            if audio_path:
                # Check if we already have the duration for this song_id
                cache_flag = False
                if song_id in song_duration_cache:
                    duration = song_duration_cache[song_id]
                    cache_flag = True
                else:
                    # Download the audio file to get its duration
                    temp_audio_file = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio_file:
                            urllib.request.urlretrieve(audio_path, temp_audio_file.name)
                            # Load the audio file to get its sample rate and duration
                            waveform, sample_rate = torchaudio.load(temp_audio_file.name)
                            duration = waveform.shape[1] / sample_rate  # Duration in seconds
                            
                            # Cache the duration for future use
                            song_duration_cache[song_id] = duration
                    except Exception as e:
                        print(f"Warning: Could not process audio file for participant {participant_id}, song {song_id}: {e}")
                        return None
                    finally:
                        # Clean up the temporary audio file
                        if temp_audio_file and os.path.exists(temp_audio_file.name):
                            os.unlink(temp_audio_file.name)
                
                result = {
                    'subject': participant_id,
                    'song_id': song_id,
                    'audio_path': audio_path,
                    'eda_path': eda_s3_path,
                    'duration': duration
                }
                cache_status = "(cache hit)" if cache_flag else "(downloaded)"
                print(f"Processed: Participant {participant_id}, Song {song_id}, Duration: {duration:.2f}s {cache_status}")
                return result
        except Exception as e:
            print(f"Warning: EDA file not found for participant {participant_id}, song {song_id}: {e}")
        
        return None
    
    # Create a list of tasks to process
    tasks = []
    
    # Process each participant and song combination
    print("Preparing participant and song combinations...")
    
    # Get unique participant IDs and song IDs
    unique_participants = av_ratings_df['participant_id'].unique()
    
    for participant_id in unique_participants:
        # Get all songs for this participant
        participant_data = av_ratings_df[av_ratings_df['participant_id'] == participant_id]
        
        for _, row in participant_data.iterrows():
            song_id = row['song_id']
            song_no = row['song_no']
            audio_path = song_id_to_audio.get(song_id)
            
            if audio_path:
                tasks.append((participant_id, song_id, song_no, audio_path))
    
    # Process tasks in parallel
    metadata_rows = []
    start_time = time.time()
    print(f"Processing {len(tasks)} tasks with multithreading...")
    
    # Use ThreadPoolExecutor to process tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_participant_song, participant_id, song_id, song_no, audio_path, s3_manager, song_duration_cache): 
            (participant_id, song_id) 
            for participant_id, song_id, song_no, audio_path in tasks
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            participant_id, song_id = future_to_task[future]
            try:
                result = future.result()
                if result:
                    metadata_rows.append(result)
            except Exception as e:
                print(f"Error processing participant {participant_id}, song {song_id}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Multithreaded processing completed in {elapsed_time:.2f} seconds")
    print(f"Successfully processed {len(metadata_rows)} out of {len(tasks)} tasks")
    
    # Create a temporary CSV file
    print(f"Writing {len(metadata_rows)} rows to CSV...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        fieldnames = ['subject', 'song_id', 'audio_path', 'eda_path', 'duration']
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
        temp_file_path = temp_file.name
    
    # Upload the file to S3
    print("Uploading to S3...")
    bucket, _ = s3_manager._parse_s3_path("s3://audio2biosignal-train-data/HKU956/")
    s3_client.upload_file(
        temp_file_path, 
        bucket, 
        "HKU956/custom_metadata.csv"
    )
    
    # Clean up the temporary file
    os.unlink(temp_file_path)
    
    print("Custom metadata file created and uploaded successfully!")
    print(f"S3 Path: s3://audio2biosignal-train-data/HKU956/custom_metadata.csv")

if __name__ == "__main__":
    main()

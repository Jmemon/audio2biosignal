import boto3
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
import os

class S3FileManager:
    def __init__(self):
        # Initialize a new instance with its own temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())

    def _get_s3_client(self):
        """Create a new S3 client on demand"""
        return boto3.client('s3')
        
    def _parse_s3_path(self, s3_path: str):
        assert s3_path.startswith('s3://'), f"Invalid S3 path: {s3_path}"
        path_parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        return bucket, key

    def download_file(self, s3_path: str) -> Path:
        bucket, key = self._parse_s3_path(s3_path)
        local_filename = key.replace('/', '_')
        local_path = self.temp_dir / local_filename

        # Create a client for this download and then discard it
        s3_client = self._get_s3_client()
        # Download the file
        s3_client.download_file(bucket, key, str(local_path))
        
        return local_path


        
    def __del__(self):
        """Clean up resources when the instance is garbage collected"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore errors during cleanup

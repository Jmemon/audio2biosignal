import boto3
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
import os

class S3FileManager:
    def __init__(self):
        """
        Initialize a new S3FileManager instance with isolated temporary storage.
        
        Creates a dedicated temporary directory for this instance to store downloaded
        S3 files, ensuring isolation between different manager instances. The directory
        is automatically cleaned up when the instance is garbage collected.
        
        Architecture:
            - Uses system temp directory as parent location
            - Creates unique subdirectory per instance for file isolation
            - O(1) initialization complexity
        
        Thread Safety:
            - Each instance maintains its own isolated temp directory
            - No shared state between instances
            
        Resource Management:
            - Temp directory automatically cleaned up during garbage collection
            - S3 clients created on-demand to minimize resource consumption
        
        Limitations:
            - Relies on garbage collection for cleanup which is non-deterministic
            - No explicit control over temp directory location
        """
        # Initialize a new instance with its own temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())

    def _get_s3_client(self):
        """Create a new S3 client on demand"""
        return boto3.client('s3')
        
    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """
        Parse an S3 URI into bucket and key components with O(1) complexity.
        
        Decomposes a fully-qualified S3 path (s3://bucket/key) into its constituent
        parts for use with boto3 operations. Performs validation to ensure the path
        conforms to the required s3:// scheme.
        
        Architecture:
            - Single-pass string parsing with constant-time operations
            - No regex for performance optimization
            - Handles edge cases like empty keys and bucket-only URIs
        
        Parameters:
            s3_path: str - A fully-qualified S3 URI in the format 's3://bucket/key'
                           The key portion may be empty for bucket-only operations
        
        Returns:
            tuple[str, str] - A tuple containing (bucket_name, object_key)
                              For bucket-only paths, object_key will be an empty string
        
        Raises:
            AssertionError - If s3_path doesn't start with 's3://' prefix
            
        Thread Safety:
            - Stateless operation, thread-safe
            
        Limitations:
            - Does not validate bucket name against S3 naming rules
            - Does not handle URL-encoded characters in paths
            - No support for S3 access points or other advanced addressing
        """
        assert s3_path.startswith('s3://'), f"Invalid S3 path: {s3_path}"
        path_parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        return bucket, key

    def download_file(self, s3_path: str) -> Path:
        """
        Download a file from S3 to a local temporary directory with path flattening.
        
        Retrieves an object from S3 and stores it in the instance's dedicated temporary
        directory with a flattened path structure. Handles the entire download lifecycle
        including path parsing, local filename generation, and S3 client management.
        
        Architecture:
            - O(1) path parsing and local filename generation
            - O(n) download complexity where n is file size
            - Path flattening replaces directory separators with underscores
            - On-demand S3 client creation for connection pooling optimization
        
        Parameters:
            s3_path: str - A fully-qualified S3 URI in the format 's3://bucket/key'
                           Must conform to S3 path requirements
        
        Returns:
            Path - Local filesystem path to the downloaded file
                   Path is within the instance's temporary directory
        
        Raises:
            AssertionError - If s3_path doesn't start with 's3://' prefix
            ClientError - If S3 access fails (permissions, no such key, etc.)
            
        Thread Safety:
            - Thread-safe for different s3_paths
            - Non-atomic for identical s3_paths (last write wins)
            
        Limitations:
            - Path flattening may cause filename collisions for different S3 paths
            - No retry mechanism for failed downloads
            - No streaming support for large files
            - Downloaded files persist until instance is garbage collected
        """
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

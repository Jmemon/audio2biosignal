import boto3
import threading
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

class S3FileManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, max_cache_size=100):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(S3FileManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def _init(self, max_cache_size):
        self.s3_client = boto3.client('s3')
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.max_cache_size = max_cache_size
        self.temp_dir = Path(tempfile.mkdtemp())
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.in_progress_downloads = {}
        self._initialized = True

    def __init__(self, max_cache_size=100):
        if not self._initialized:
            self._init(max_cache_size)

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

        with self.cache_lock:
            if s3_path in self.cache:
                # Move to end to show that it was recently used
                self.cache.move_to_end(s3_path)
                return self.cache[s3_path]
            elif s3_path in self.in_progress_downloads:
                # Wait for the in-progress download to finish
                self.in_progress_downloads[s3_path].result()
                return self.cache[s3_path]

        # Download the file
        self.s3_client.download_file(bucket, key, str(local_path))

        with self.cache_lock:
            self.cache[s3_path] = local_path
            if len(self.cache) > self.max_cache_size:
                # Remove the least recently used item
                old_s3_path, old_local_path = self.cache.popitem(last=False)
                old_local_path.unlink()
        return local_path

    def prefetch_files(self, s3_paths: List[str]) -> None:
        for s3_path in s3_paths:
            with self.cache_lock:
                if s3_path in self.cache or s3_path in self.in_progress_downloads:
                    continue
                future = self.executor.submit(self.download_file, s3_path)
                self.in_progress_downloads[s3_path] = future
                future.add_done_callback(self._download_complete(s3_path))

    def _download_complete(self, s3_path):
        def _callback(future):
            with self.cache_lock:
                if future.cancelled():
                    return
                self.in_progress_downloads.pop(s3_path, None)
        return _callback

    def get_file(self, s3_path: str) -> Optional[Path]:
        with self.cache_lock:
            if s3_path in self.cache:
                self.cache.move_to_end(s3_path)
                return self.cache[s3_path]
            return None

    def clear_cache(self) -> None:
        with self.cache_lock:
            for local_path in self.cache.values():
                local_path.unlink()
            self.cache.clear()
        shutil.rmtree(self.temp_dir)
        self.temp_dir = Path(tempfile.mkdtemp())

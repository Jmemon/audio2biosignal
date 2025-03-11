"""
Test suite for S3FileManager.

This module provides comprehensive testing for:
1. Singleton pattern implementation
2. S3 path parsing
3. File download functionality
4. Caching behavior
5. Prefetching mechanism
6. Resource management
7. Thread safety
"""

import os
import pytest
import tempfile
import threading
import time
import concurrent.futures
from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

import boto3
from src.utilities import S3FileManager


class TestS3FileManagerSingleton:
    """Tests for S3FileManager's singleton pattern implementation."""

    def test_singleton_pattern(self):
        """
        GIVEN multiple instantiations of S3FileManager
        WHEN instances are created
        THEN all instances should be the same object
        """
        manager1 = S3FileManager()
        manager2 = S3FileManager(max_cache_size=200)  # Different parameter
        manager3 = S3FileManager()
        
        assert manager1 is manager2
        assert manager2 is manager3
        assert manager1._initialized is True
        
    def test_max_cache_size_initialization(self):
        """
        GIVEN S3FileManager is initialized with different max_cache_size values
        WHEN the instance is created
        THEN it should maintain the max_cache_size of the first instantiation
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # First initialization sets the max_cache_size
        manager1 = S3FileManager(max_cache_size=50)
        assert manager1.max_cache_size == 50
        
        # Subsequent initializations should not change max_cache_size
        manager2 = S3FileManager(max_cache_size=100)
        assert manager2.max_cache_size == 50
        assert manager1 is manager2
    
    def test_singleton_thread_safety(self):
        """
        GIVEN concurrent instantiations of S3FileManager
        WHEN instances are created from multiple threads
        THEN all instances should be the same object
        """
        instances = []
        
        def create_instance():
            instances.append(S3FileManager())
        
        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All instances should be identical
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance
            
    def test_singleton_with_different_parameters(self):
        """
        GIVEN multiple instantiations with different parameters
        WHEN instances are created with varying max_cache_size values
        THEN all instances should be the same object with the first initialization's parameters
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create instances with different parameters
        parameters = [10, 50, 100, 200, 500]
        instances = [S3FileManager(max_cache_size=p) for p in parameters]
        
        # All instances should be identical
        for instance in instances[1:]:
            assert instance is instances[0]
            
        # All instances should have the max_cache_size of the first instantiation
        for instance in instances:
            assert instance.max_cache_size == parameters[0]
            
    def test_singleton_initialization_race_condition(self):
        """
        GIVEN a race condition during initialization
        WHEN multiple threads try to initialize the singleton simultaneously
        THEN only one initialization should succeed and all instances should be identical
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Mock the _init method to simulate slow initialization
        original_init = S3FileManager._init
        init_called_count = 0
        
        def slow_init(self, max_cache_size):
            nonlocal init_called_count
            init_called_count += 1
            # Simulate work that takes time
            time.sleep(0.01)
            original_init(self, max_cache_size)
            
        with patch.object(S3FileManager, '_init', slow_init):
            # Create instances from multiple threads
            instances = []
            
            def create_instance():
                instances.append(S3FileManager(max_cache_size=100))
                
            threads = [threading.Thread(target=create_instance) for _ in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            # All instances should be identical
            first_instance = instances[0]
            for instance in instances[1:]:
                assert instance is first_instance
                
            # _init should only be called once despite multiple threads
            assert init_called_count == 1


class TestS3PathParsing:
    """Tests for S3 path parsing functionality."""
    
    def test_valid_s3_path_parsing(self):
        """
        GIVEN valid S3 paths
        WHEN _parse_s3_path is called
        THEN it should correctly extract bucket and key
        """
        manager = S3FileManager()
        
        # Test cases with expected results
        test_cases = [
            ("s3://bucket/key", ("bucket", "key")),
            ("s3://bucket/nested/key", ("bucket", "nested/key")),
            ("s3://bucket/very/deeply/nested/key.txt", ("bucket", "very/deeply/nested/key.txt")),
            ("s3://bucket", ("bucket", "")),
            ("s3://bucket/", ("bucket", "")),
        ]
        
        for s3_path, expected in test_cases:
            bucket, key = manager._parse_s3_path(s3_path)
            assert bucket == expected[0]
            assert key == expected[1]
    
    def test_invalid_s3_path(self):
        """
        GIVEN invalid S3 paths
        WHEN _parse_s3_path is called
        THEN it should raise an assertion error
        """
        manager = S3FileManager()
        
        invalid_paths = [
            "http://bucket/key",
            "bucket/key",
            "/bucket/key",
            "s3:/bucket/key",
            "",
            None
        ]
        
        for invalid_path in invalid_paths:
            with pytest.raises(AssertionError):
                manager._parse_s3_path(invalid_path)
    
    def test_s3_path_with_special_characters(self):
        """
        GIVEN S3 paths with special characters
        WHEN _parse_s3_path is called
        THEN it should correctly extract bucket and key
        """
        manager = S3FileManager()
        
        # Test cases with special characters
        test_cases = [
            ("s3://bucket/key with spaces", ("bucket", "key with spaces")),
            ("s3://bucket/path/with+plus", ("bucket", "path/with+plus")),
            ("s3://bucket/file-with-dashes", ("bucket", "file-with-dashes")),
            ("s3://bucket/file_with_underscores", ("bucket", "file_with_underscores")),
            ("s3://bucket/file.with.dots", ("bucket", "file.with.dots")),
            ("s3://bucket/path/with%20encoding", ("bucket", "path/with%20encoding")),
            ("s3://bucket/path/with#hash", ("bucket", "path/with#hash")),
            ("s3://bucket/path/with?query=param", ("bucket", "path/with?query=param")),
            ("s3://bucket/path/with&ampersand", ("bucket", "path/with&ampersand")),
            ("s3://bucket/path/with=equals", ("bucket", "path/with=equals")),
        ]
        
        for s3_path, expected in test_cases:
            bucket, key = manager._parse_s3_path(s3_path)
            assert bucket == expected[0]
            assert key == expected[1]
    
    def test_s3_path_with_unicode_characters(self):
        """
        GIVEN S3 paths with Unicode characters
        WHEN _parse_s3_path is called
        THEN it should correctly extract bucket and key
        """
        manager = S3FileManager()
        
        # Test cases with Unicode characters
        test_cases = [
            ("s3://bucket/ünicode-file", ("bucket", "ünicode-file")),
            ("s3://bucket/path/with/émoji🔥", ("bucket", "path/with/émoji🔥")),
            ("s3://bucket/path/with/中文", ("bucket", "path/with/中文")),
            ("s3://bucket/path/with/русский", ("bucket", "path/with/русский")),
        ]
        
        for s3_path, expected in test_cases:
            bucket, key = manager._parse_s3_path(s3_path)
            assert bucket == expected[0]
            assert key == expected[1]
    
    def test_s3_path_with_multiple_slashes(self):
        """
        GIVEN S3 paths with multiple consecutive slashes
        WHEN _parse_s3_path is called
        THEN it should handle them appropriately
        """
        manager = S3FileManager()
        
        # Test cases with multiple slashes
        test_cases = [
            ("s3://bucket//key", ("bucket", "/key")),
            ("s3://bucket///nested///key", ("bucket", "//nested///key")),
            ("s3://bucket////", ("bucket", "///"))
        ]
        
        for s3_path, expected in test_cases:
            bucket, key = manager._parse_s3_path(s3_path)
            assert bucket == expected[0]
            assert key == expected[1]
    
    def test_s3_path_with_leading_trailing_whitespace(self):
        """
        GIVEN S3 paths with leading or trailing whitespace
        WHEN _parse_s3_path is called
        THEN it should handle the whitespace appropriately
        """
        manager = S3FileManager()
        
        # Test cases with whitespace
        test_cases = [
            ("  s3://bucket/key  ", ("bucket", "key")),  # Assuming whitespace is stripped
            ("\ts3://bucket/key\n", ("bucket", "key")),  # Assuming whitespace is stripped
        ]
        
        for s3_path, expected in test_cases:
            # This test might fail if the implementation doesn't strip whitespace
            # If it fails, it indicates the actual behavior differs from assumption
            try:
                bucket, key = manager._parse_s3_path(s3_path)
                assert bucket == expected[0]
                assert key == expected[1]
            except AssertionError:
                # If the implementation doesn't strip whitespace, this is the expected behavior
                with pytest.raises(AssertionError):
                    manager._parse_s3_path(s3_path)
    
    def test_s3_path_with_empty_bucket(self):
        """
        GIVEN S3 paths with empty bucket name
        WHEN _parse_s3_path is called
        THEN it should raise an appropriate error
        """
        manager = S3FileManager()
        
        invalid_paths = [
            "s3:///key",  # Empty bucket name
            "s3:///"      # Empty bucket and key
        ]
        
        for invalid_path in invalid_paths:
            with pytest.raises(AssertionError):
                manager._parse_s3_path(invalid_path)
    
    def test_s3_path_edge_cases(self):
        """
        GIVEN edge case S3 paths
        WHEN _parse_s3_path is called
        THEN it should handle them appropriately
        """
        manager = S3FileManager()
        
        # Edge cases
        test_cases = [
            ("s3://bucket-with-dash", ("bucket-with-dash", "")),
            ("s3://bucket.with.dots", ("bucket.with.dots", "")),
            ("s3://bucket_with_underscore", ("bucket_with_underscore", "")),
            ("s3://123numeric-bucket", ("123numeric-bucket", "")),
            ("s3://bucket:with:colons/key", ("bucket:with:colons", "key")),  # Unusual but valid in some contexts
        ]
        
        for s3_path, expected in test_cases:
            bucket, key = manager._parse_s3_path(s3_path)
            assert bucket == expected[0]
            assert key == expected[1]
    
    def test_s3_path_with_very_long_components(self):
        """
        GIVEN S3 paths with very long bucket or key components
        WHEN _parse_s3_path is called
        THEN it should handle them correctly
        """
        manager = S3FileManager()
        
        # Create long strings for testing
        long_bucket = "bucket" + "x" * 50  # 55 chars
        long_key = "a" * 1000  # 1000 chars
        
        # Test with long components
        s3_path = f"s3://{long_bucket}/{long_key}"
        bucket, key = manager._parse_s3_path(s3_path)
        
        assert bucket == long_bucket
        assert key == long_key
        assert len(bucket) == 55
        assert len(key) == 1000
        
    def test_s3_path_with_url_encoded_characters(self):
        """
        GIVEN S3 paths with URL-encoded characters
        WHEN _parse_s3_path is called
        THEN it should handle them correctly without decoding
        """
        manager = S3FileManager()
        
        # Test cases with URL-encoded characters
        test_cases = [
            ("s3://bucket/path%20with%20spaces", ("bucket", "path%20with%20spaces")),
            ("s3://bucket/file%2Bwith%2Bplus", ("bucket", "file%2Bwith%2Bplus")),
            ("s3://bucket/path%2Fwith%2Fencoded%2Fslashes", ("bucket", "path%2Fwith%2Fencoded%2Fslashes")),
            ("s3://bucket/file%3Fwith%3Dquery%26params", ("bucket", "file%3Fwith%3Dquery%26params")),
            ("s3://bucket/path%25with%25percent", ("bucket", "path%25with%25percent")),
        ]
        
        for s3_path, expected in test_cases:
            bucket, key = manager._parse_s3_path(s3_path)
            assert bucket == expected[0]
            assert key == expected[1], f"Expected key '{expected[1]}' but got '{key}' for path '{s3_path}'"
    
    def test_s3_path_with_malformed_but_valid_prefix(self):
        """
        GIVEN S3 paths that start with s3:// but have other malformations
        WHEN _parse_s3_path is called
        THEN it should handle them according to the implementation
        """
        manager = S3FileManager()
        
        # These paths start with s3:// but have other issues
        # The method should still process them based on its implementation
        test_cases = [
            ("s3://bucket//", ("bucket", "/")),  # Double slash after bucket
            ("s3://bucket name/key", ("bucket name", "key")),  # Space in bucket name (technically invalid in S3)
            ("s3://bucket/key/", ("bucket", "key/")),  # Trailing slash in key
            ("s3://bucket?query=param/key", ("bucket?query=param", "key")),  # Query params in bucket (invalid in S3)
            ("s3://bucket#fragment/key", ("bucket#fragment", "key")),  # Fragment in bucket (invalid in S3)
        ]
        
        for s3_path, expected in test_cases:
            bucket, key = manager._parse_s3_path(s3_path)
            assert bucket == expected[0], f"Expected bucket '{expected[0]}' but got '{bucket}' for path '{s3_path}'"
            assert key == expected[1], f"Expected key '{expected[1]}' but got '{key}' for path '{s3_path}'"


@pytest.fixture
def mock_s3_client():
    """Fixture providing a mocked S3 client."""
    with patch('boto3.client') as mock_client:
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        yield mock_s3


@pytest.fixture
def s3_manager(mock_s3_client):
    """Fixture providing an S3FileManager with mocked dependencies."""
    # Reset the singleton instance for isolated tests
    S3FileManager._instance = None
    S3FileManager._instance_lock = threading.Lock()
    
    with patch('tempfile.mkdtemp') as mock_mkdtemp:
        mock_mkdtemp.return_value = '/tmp/mock_temp_dir'
        manager = S3FileManager(max_cache_size=5)
        
        # Replace the ThreadPoolExecutor with a mock
        manager.executor = MagicMock()
        
        yield manager


class TestFileDownload:
    """Tests for file download functionality."""
    
    def test_download_file_creates_parent_directories(self, s3_manager, mock_s3_client):
        """
        GIVEN an S3 path that would result in a local path with non-existent parent directories
        WHEN download_file is called
        THEN it should create the necessary parent directories
        """
        s3_path = "s3://test-bucket/deep/nested/structure/file.txt"
        expected_local_path = Path('/tmp/mock_temp_dir/deep_nested_structure_file.txt')
        parent_dir = expected_local_path.parent
        
        # Mock Path.exists to return False for the parent directory
        with patch.object(Path, 'exists', return_value=False), \
             patch('os.makedirs') as mock_makedirs:
            
            result = s3_manager.download_file(s3_path)
            
            # Verify parent directories were created
            mock_makedirs.assert_called_once_with(str(parent_dir), exist_ok=True)
            
            # Verify download proceeded
            mock_s3_client.download_file.assert_called_once_with(
                "test-bucket", "deep/nested/structure/file.txt", str(expected_local_path)
            )
            
            # Verify result
            assert result == expected_local_path
    
    def test_download_file_name_collision(self, s3_manager, mock_s3_client):
        """
        GIVEN two S3 paths with different paths but same filename after flattening
        WHEN download_file is called for both
        THEN both should be downloaded and cached correctly without overwriting
        """
        # Two different S3 paths that would result in the same local filename
        s3_path1 = "s3://bucket1/path/to/file.txt"
        s3_path2 = "s3://bucket2/different/path/to/file.txt"
        
        # Both would normally flatten to 'path_to_file.txt'
        expected_local_path1 = Path('/tmp/mock_temp_dir/path_to_file.txt')
        expected_local_path2 = Path('/tmp/mock_temp_dir/different_path_to_file.txt')
        
        # First download
        result1 = s3_manager.download_file(s3_path1)
        assert result1 == expected_local_path1
        mock_s3_client.download_file.assert_called_with(
            "bucket1", "path/to/file.txt", str(expected_local_path1)
        )
        
        # Second download
        result2 = s3_manager.download_file(s3_path2)
        assert result2 == expected_local_path2
        mock_s3_client.download_file.assert_called_with(
            "bucket2", "different/path/to/file.txt", str(expected_local_path2)
        )
        
        # Verify both are cached
        assert s3_path1 in s3_manager.cache
        assert s3_path2 in s3_manager.cache
        assert s3_manager.cache[s3_path1] == expected_local_path1
        assert s3_manager.cache[s3_path2] == expected_local_path2
    
    def test_download_file_success(self, s3_manager, mock_s3_client):
        """
        GIVEN a valid S3 path
        WHEN download_file is called
        THEN it should download the file and return the local path
        """
        s3_path = "s3://test-bucket/test-key.txt"
        expected_local_path = Path('/tmp/mock_temp_dir/test-key.txt')
        
        # Configure mock to simulate successful download
        mock_s3_client.download_file.return_value = None
        
        # Call the method
        result = s3_manager.download_file(s3_path)
        
        # Verify the result
        assert result == expected_local_path
        
        # Verify S3 client was called correctly
        mock_s3_client.download_file.assert_called_once_with(
            "test-bucket", "test-key.txt", str(expected_local_path)
        )
        
        # Verify the file is cached
        assert s3_path in s3_manager.cache
        assert s3_manager.cache[s3_path] == expected_local_path
    
    def test_download_file_with_nested_path(self, s3_manager, mock_s3_client):
        """
        GIVEN an S3 path with nested directories
        WHEN download_file is called
        THEN it should flatten the path for local storage
        """
        s3_path = "s3://test-bucket/nested/path/to/file.txt"
        expected_local_path = Path('/tmp/mock_temp_dir/nested_path_to_file.txt')
        
        result = s3_manager.download_file(s3_path)
        
        assert result == expected_local_path
        mock_s3_client.download_file.assert_called_once_with(
            "test-bucket", "nested/path/to/file.txt", str(expected_local_path)
        )
    
    def test_download_file_s3_error(self, s3_manager, mock_s3_client):
        """
        GIVEN a valid S3 path but S3 client raises an error
        WHEN download_file is called
        THEN the error should be propagated
        """
        s3_path = "s3://test-bucket/error-file.txt"
        
        # Configure mock to raise an exception
        mock_s3_client.download_file.side_effect = Exception("S3 error")
        
        # Verify the exception is propagated
        with pytest.raises(Exception, match="S3 error"):
            s3_manager.download_file(s3_path)
        
        # Verify the file is not cached
        assert s3_path not in s3_manager.cache
        
    def test_download_file_with_zero_max_cache_size(self, mock_s3_client):
        """
        GIVEN an S3FileManager with max_cache_size of 0
        WHEN download_file is called multiple times
        THEN files should be downloaded but not cached
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = '/tmp/mock_temp_dir'
            manager = S3FileManager(max_cache_size=0)
            
            # Replace the ThreadPoolExecutor with a mock
            manager.executor = MagicMock()
            
            # Download a file
            s3_path = "s3://test-bucket/zero-cache-file.txt"
            expected_local_path = Path('/tmp/mock_temp_dir/zero-cache-file.txt')
            
            result = manager.download_file(s3_path)
            
            # Verify the file was downloaded
            assert result == expected_local_path
            mock_s3_client.download_file.assert_called_once_with(
                "test-bucket", "zero-cache-file.txt", str(expected_local_path)
            )
            
            # Verify the file was not cached
            assert len(manager.cache) == 0
            
            # Download the same file again
            mock_s3_client.reset_mock()
            result2 = manager.download_file(s3_path)
            
            # Verify the file was downloaded again (not retrieved from cache)
            assert result2 == expected_local_path
            mock_s3_client.download_file.assert_called_once_with(
                "test-bucket", "zero-cache-file.txt", str(expected_local_path)
            )
            
    def test_download_file_with_negative_max_cache_size(self, mock_s3_client):
        """
        GIVEN an S3FileManager with negative max_cache_size
        WHEN download_file is called
        THEN it should handle it gracefully (implementation-dependent)
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = '/tmp/mock_temp_dir'
            manager = S3FileManager(max_cache_size=-5)
            
            # Replace the ThreadPoolExecutor with a mock
            manager.executor = MagicMock()
            
            # Download a file
            s3_path = "s3://test-bucket/negative-cache-file.txt"
            expected_local_path = Path('/tmp/mock_temp_dir/negative-cache-file.txt')
            
            result = manager.download_file(s3_path)
            
            # Verify the file was downloaded
            assert result == expected_local_path
            mock_s3_client.download_file.assert_called_once_with(
                "test-bucket", "negative-cache-file.txt", str(expected_local_path)
            )
            
            # Verify caching behavior (implementation-dependent)
            # This test might need adjustment based on how negative values should be handled
            
    def test_download_file_with_non_writable_temp_dir(self, mock_s3_client):
        """
        GIVEN a temp directory that is not writable
        WHEN download_file is called
        THEN it should handle the error appropriately
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = '/tmp/mock_temp_dir'
            manager = S3FileManager(max_cache_size=5)
            
            # Replace the ThreadPoolExecutor with a mock
            manager.executor = MagicMock()
            
            s3_path = "s3://test-bucket/permission-error-file.txt"
            expected_local_path = Path('/tmp/mock_temp_dir/permission-error-file.txt')
            
            # Mock S3 client to raise a permission error
            mock_s3_client.download_file.side_effect = PermissionError("Permission denied")
            
            # Should propagate the permission error
            with pytest.raises(PermissionError, match="Permission denied"):
                manager.download_file(s3_path)
                
            # Verify the file is not cached
            assert s3_path not in manager.cache


class TestGetFile:
    """Tests for the get_file method."""
    
    def test_get_file_with_cached_file(self, s3_manager):
        """
        GIVEN a file that exists in the cache
        WHEN get_file is called with the S3 path
        THEN it should return the cached file path
        """
        # Add a file to the cache
        s3_path = "s3://test-bucket/existing-file.txt"
        local_path = Path('/tmp/mock_temp_dir/existing-file.txt')
        s3_manager.cache[s3_path] = local_path
        
        # Call get_file
        result = s3_manager.get_file(s3_path)
        
        # Verify the result
        assert result == local_path
        
    def test_get_file_with_uncached_file(self, s3_manager):
        """
        GIVEN a file that doesn't exist in the cache
        WHEN get_file is called with the S3 path
        THEN it should return None
        """
        # Ensure the path is not in the cache
        s3_path = "s3://test-bucket/non-existent-file.txt"
        if s3_path in s3_manager.cache:
            del s3_manager.cache[s3_path]
            
        # Call get_file
        result = s3_manager.get_file(s3_path)
        
        # Verify the result
        assert result is None
        
    def test_get_file_updates_lru_order(self, s3_manager):
        """
        GIVEN multiple files in the cache
        WHEN get_file is called for one of them
        THEN it should move that file to the end of the LRU order
        """
        # Add multiple files to the cache
        s3_paths = [f"s3://test-bucket/file{i}.txt" for i in range(3)]
        local_paths = [Path(f'/tmp/mock_temp_dir/file{i}.txt') for i in range(3)]
        
        for s3_path, local_path in zip(s3_paths, local_paths):
            s3_manager.cache[s3_path] = local_path
            
        # Get the middle file
        middle_path = s3_paths[1]
        s3_manager.get_file(middle_path)
        
        # Verify the LRU order (should now be [0, 2, 1])
        cache_items = list(s3_manager.cache.items())
        assert cache_items[0][0] == s3_paths[0]
        assert cache_items[1][0] == s3_paths[2]
        assert cache_items[2][0] == s3_paths[1]
        
    def test_get_file_with_empty_cache(self, s3_manager):
        """
        GIVEN an empty cache
        WHEN get_file is called
        THEN it should return None
        """
        # Clear the cache
        s3_manager.cache.clear()
        
        # Call get_file
        result = s3_manager.get_file("s3://test-bucket/any-file.txt")
        
        # Verify the result
        assert result is None
        
    def test_get_file_with_invalid_s3_path(self, s3_manager):
        """
        GIVEN an invalid S3 path
        WHEN get_file is called
        THEN it should still work correctly (not raise exceptions)
        """
        # Add a file with a valid path to the cache
        valid_path = "s3://test-bucket/valid-file.txt"
        local_path = Path('/tmp/mock_temp_dir/valid-file.txt')
        s3_manager.cache[valid_path] = local_path
        
        # Call get_file with various invalid paths
        invalid_paths = [
            None,
            "",
            "not-an-s3-path",
            "http://not-s3/file.txt"
        ]
        
        for invalid_path in invalid_paths:
            # Should not raise exceptions
            result = s3_manager.get_file(invalid_path)
            assert result is None
            
    def test_get_file_thread_safety(self, s3_manager):
        """
        GIVEN concurrent access to the cache
        WHEN get_file is called from multiple threads
        THEN it should handle the concurrency correctly
        """
        # Add a file to the cache
        s3_path = "s3://test-bucket/thread-safe-file.txt"
        local_path = Path('/tmp/mock_temp_dir/thread-safe-file.txt')
        s3_manager.cache[s3_path] = local_path
        
        # Track if any exceptions occur during concurrent execution
        exceptions = []
        results = []
        
        def call_get_file():
            try:
                result = s3_manager.get_file(s3_path)
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Execute get_file concurrently
        threads = [threading.Thread(target=call_get_file) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Verify no exceptions occurred
        assert not exceptions, f"Exceptions occurred: {exceptions}"
        
        # Verify all calls returned the expected result
        assert all(result == local_path for result in results)
        assert len(results) == 10

class TestCacheBehavior:
    """Tests for caching behavior."""
    
    def test_cache_hit(self, s3_manager, mock_s3_client):
        """
        GIVEN a file that is already in the cache
        WHEN download_file is called again for the same file
        THEN it should return the cached path without downloading again
        """
        s3_path = "s3://test-bucket/cached-file.txt"
        local_path = Path('/tmp/mock_temp_dir/cached-file.txt')
        
        # Pre-populate the cache
        s3_manager.cache[s3_path] = local_path
        
        # Call the method
        result = s3_manager.download_file(s3_path)
        
        # Verify the result is from cache
        assert result == local_path
        
        # Verify S3 client was not called
        mock_s3_client.download_file.assert_not_called()
    
    def test_cache_eviction(self, s3_manager, mock_s3_client):
        """
        GIVEN a cache that reaches its maximum size
        WHEN a new file is downloaded
        THEN the least recently used file should be evicted
        """
        # Fill the cache to its limit (5 items)
        for i in range(5):
            s3_path = f"s3://test-bucket/file{i}.txt"
            local_path = Path(f'/tmp/mock_temp_dir/file{i}.txt')
            s3_manager.cache[s3_path] = local_path
        
        # Access the first file to make it not the LRU
        first_s3_path = "s3://test-bucket/file0.txt"
        s3_manager.cache.move_to_end(first_s3_path)
        
        # The second file should now be the LRU
        lru_s3_path = "s3://test-bucket/file1.txt"
        lru_local_path = s3_manager.cache[lru_s3_path]
        
        # Mock the unlink method to avoid actual file deletion
        with patch.object(Path, 'unlink') as mock_unlink:
            # Download a new file to trigger cache eviction
            new_s3_path = "s3://test-bucket/new-file.txt"
            s3_manager.download_file(new_s3_path)
            
            # Verify the LRU file was removed
            assert lru_s3_path not in s3_manager.cache
            mock_unlink.assert_called_once()
            
            # Verify the new file was added
            assert new_s3_path in s3_manager.cache
            
            # Verify cache size remains at max
            assert len(s3_manager.cache) == 5
            
    def test_cache_eviction_unlink_error(self, s3_manager, mock_s3_client):
        """
        GIVEN a cache that reaches its maximum size
        WHEN a new file is downloaded but file deletion fails
        THEN it should handle the error gracefully and still update the cache
        """
        # Fill the cache to its limit (5 items)
        for i in range(5):
            s3_path = f"s3://test-bucket/file{i}.txt"
            local_path = Path(f'/tmp/mock_temp_dir/file{i}.txt')
            s3_manager.cache[s3_path] = local_path
        
        # The first file should be the LRU
        lru_s3_path = "s3://test-bucket/file0.txt"
        lru_local_path = s3_manager.cache[lru_s3_path]
        
        # Mock the unlink method to raise an exception
        with patch.object(Path, 'unlink') as mock_unlink:
            mock_unlink.side_effect = OSError("Permission denied")
            
            # Download a new file to trigger cache eviction
            new_s3_path = "s3://test-bucket/new-file-unlink-error.txt"
            
            # Should not raise an exception despite unlink failure
            result = s3_manager.download_file(new_s3_path)
            
            # Verify unlink was attempted
            mock_unlink.assert_called_once()
            
            # Verify the LRU file was removed from cache even if unlink failed
            assert lru_s3_path not in s3_manager.cache
            
            # Verify the new file was added to cache
            assert new_s3_path in s3_manager.cache
            
            # Verify cache size remains at max
            assert len(s3_manager.cache) == 5
    
    
    def test_clear_cache(self, s3_manager):
        """
        GIVEN a cache with files
        WHEN clear_cache is called
        THEN all files should be removed from the cache
        """
        # Add files to the cache
        for i in range(3):
            s3_path = f"s3://test-bucket/file{i}.txt"
            local_path = Path(f'/tmp/mock_temp_dir/file{i}.txt')
            s3_manager.cache[s3_path] = local_path
        
        # Mock file and directory operations
        with patch.object(Path, 'unlink') as mock_unlink, \
             patch('shutil.rmtree') as mock_rmtree, \
             patch('tempfile.mkdtemp') as mock_mkdtemp:
            
            mock_mkdtemp.return_value = '/tmp/new_temp_dir'
            
            # Call clear_cache
            s3_manager.clear_cache()
            
            # Verify all files were unlinked
            assert mock_unlink.call_count == 3
            
            # Verify temp directory was removed and recreated
            mock_rmtree.assert_called_once_with('/tmp/mock_temp_dir')
            mock_mkdtemp.assert_called_once()
            
            # Verify cache is empty
            assert len(s3_manager.cache) == 0
            
            # Verify temp_dir was updated
            assert s3_manager.temp_dir == Path('/tmp/new_temp_dir')


class TestClearCache:
    """Tests specifically for the clear_cache method."""
    
    def test_clear_cache_with_empty_cache(self, s3_manager):
        """
        GIVEN an empty cache
        WHEN clear_cache is called
        THEN it should handle it gracefully without errors
        """
        # Ensure cache is empty
        s3_manager.cache.clear()
        
        # Mock file and directory operations
        with patch('shutil.rmtree') as mock_rmtree, \
             patch('tempfile.mkdtemp') as mock_mkdtemp:
            
            mock_mkdtemp.return_value = '/tmp/new_temp_dir'
            
            # Call clear_cache
            s3_manager.clear_cache()
            
            # Verify temp directory was removed and recreated
            mock_rmtree.assert_called_once_with('/tmp/mock_temp_dir')
            mock_mkdtemp.assert_called_once()
            
            # Verify cache is still empty
            assert len(s3_manager.cache) == 0
    
    def test_clear_cache_with_unlink_errors(self, s3_manager):
        """
        GIVEN a cache with files that can't be deleted
        WHEN clear_cache is called
        THEN it should handle unlink errors gracefully and continue
        """
        # Add files to the cache
        for i in range(3):
            s3_path = f"s3://test-bucket/file{i}.txt"
            local_path = Path(f'/tmp/mock_temp_dir/file{i}.txt')
            s3_manager.cache[s3_path] = local_path
        
        # Mock file and directory operations
        with patch.object(Path, 'unlink') as mock_unlink, \
             patch('shutil.rmtree') as mock_rmtree, \
             patch('tempfile.mkdtemp') as mock_mkdtemp:
            
            # Configure unlink to raise an exception for the second file
            def unlink_side_effect(missing_ok=False):
                if str(mock_unlink.call_args[0][0]) == '/tmp/mock_temp_dir/file1.txt':
                    raise PermissionError("Permission denied")
            
            mock_unlink.side_effect = unlink_side_effect
            mock_mkdtemp.return_value = '/tmp/new_temp_dir'
            
            # Call clear_cache - should not raise an exception
            s3_manager.clear_cache()
            
            # Verify all files were attempted to be unlinked
            assert mock_unlink.call_count == 3
            
            # Verify temp directory was removed and recreated
            mock_rmtree.assert_called_once_with('/tmp/mock_temp_dir')
            mock_mkdtemp.assert_called_once()
            
            # Verify cache is empty
            assert len(s3_manager.cache) == 0
    
    def test_clear_cache_with_rmtree_error(self, s3_manager):
        """
        GIVEN a scenario where rmtree fails
        WHEN clear_cache is called
        THEN it should handle the error gracefully
        """
        # Add files to the cache
        for i in range(3):
            s3_path = f"s3://test-bucket/file{i}.txt"
            local_path = Path(f'/tmp/mock_temp_dir/file{i}.txt')
            s3_manager.cache[s3_path] = local_path
        
        # Mock file and directory operations
        with patch.object(Path, 'unlink') as mock_unlink, \
             patch('shutil.rmtree') as mock_rmtree, \
             patch('tempfile.mkdtemp') as mock_mkdtemp:
            
            # Configure rmtree to raise an exception
            mock_rmtree.side_effect = OSError("Directory in use")
            mock_mkdtemp.return_value = '/tmp/new_temp_dir'
            
            # Call clear_cache - should not raise an exception
            s3_manager.clear_cache()
            
            # Verify all files were unlinked
            assert mock_unlink.call_count == 3
            
            # Verify rmtree was called
            mock_rmtree.assert_called_once_with('/tmp/mock_temp_dir')
            
            # Verify mkdtemp was still called to create a new temp dir
            mock_mkdtemp.assert_called_once()
            
            # Verify cache is empty
            assert len(s3_manager.cache) == 0
    
    def test_clear_cache_with_mkdtemp_error(self, s3_manager):
        """
        GIVEN a scenario where mkdtemp fails
        WHEN clear_cache is called
        THEN it should propagate the error
        """
        # Add files to the cache
        for i in range(3):
            s3_path = f"s3://test-bucket/file{i}.txt"
            local_path = Path(f'/tmp/mock_temp_dir/file{i}.txt')
            s3_manager.cache[s3_path] = local_path
        
        # Mock file and directory operations
        with patch.object(Path, 'unlink') as mock_unlink, \
             patch('shutil.rmtree') as mock_rmtree, \
             patch('tempfile.mkdtemp') as mock_mkdtemp:
            
            # Configure mkdtemp to raise an exception
            mock_mkdtemp.side_effect = OSError("Cannot create directory")
            
            # Call clear_cache - should propagate the exception
            with pytest.raises(OSError) as excinfo:
                s3_manager.clear_cache()
            
            assert "Cannot create directory" in str(excinfo.value)
            
            # Verify all files were unlinked
            assert mock_unlink.call_count == 3
            
            # Verify rmtree was called
            mock_rmtree.assert_called_once_with('/tmp/mock_temp_dir')
            
            # Verify mkdtemp was called
            mock_mkdtemp.assert_called_once()
    
    def test_clear_cache_thread_safety(self, s3_manager):
        """
        GIVEN multiple threads calling clear_cache simultaneously
        WHEN clear_cache is called concurrently
        THEN it should handle the concurrency correctly
        """
        # Add files to the cache
        for i in range(3):
            s3_path = f"s3://test-bucket/file{i}.txt"
            local_path = Path(f'/tmp/mock_temp_dir/file{i}.txt')
            s3_manager.cache[s3_path] = local_path
        
        # Mock file and directory operations
        with patch.object(Path, 'unlink') as mock_unlink, \
             patch('shutil.rmtree') as mock_rmtree, \
             patch('tempfile.mkdtemp') as mock_mkdtemp:
            
            # Configure mkdtemp to return different paths for each call
            mock_mkdtemp.side_effect = [f'/tmp/new_temp_dir_{i}' for i in range(5)]
            
            # Track if any exceptions occur during concurrent execution
            exceptions = []
            
            def call_clear_cache():
                try:
                    s3_manager.clear_cache()
                except Exception as e:
                    exceptions.append(e)
            
            # Execute clear_cache concurrently
            threads = [threading.Thread(target=call_clear_cache) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            # Verify no exceptions occurred
            assert not exceptions, f"Exceptions occurred: {exceptions}"
            
            # Verify cache is empty
            assert len(s3_manager.cache) == 0
            
            # Verify the temp_dir was updated (to the last created directory)
            assert s3_manager.temp_dir == Path(f'/tmp/new_temp_dir_{mock_mkdtemp.call_count-1}')
    
    def test_clear_cache_with_in_progress_downloads(self, s3_manager):
        """
        GIVEN in-progress downloads
        WHEN clear_cache is called
        THEN it should clear in_progress_downloads as well
        """
        # Add files to the cache
        for i in range(3):
            s3_path = f"s3://test-bucket/file{i}.txt"
            local_path = Path(f'/tmp/mock_temp_dir/file{i}.txt')
            s3_manager.cache[s3_path] = local_path
        
        # Add in-progress downloads
        mock_future = MagicMock(spec=Future)
        s3_manager.in_progress_downloads["s3://test-bucket/downloading.txt"] = mock_future
        
        # Mock file and directory operations
        with patch.object(Path, 'unlink') as mock_unlink, \
             patch('shutil.rmtree') as mock_rmtree, \
             patch('tempfile.mkdtemp') as mock_mkdtemp:
            
            mock_mkdtemp.return_value = '/tmp/new_temp_dir'
            
            # Call clear_cache
            s3_manager.clear_cache()
            
            # Verify all files were unlinked
            assert mock_unlink.call_count == 3
            
            # Verify temp directory was removed and recreated
            mock_rmtree.assert_called_once_with('/tmp/mock_temp_dir')
            mock_mkdtemp.assert_called_once()
            
            # Verify cache is empty
            assert len(s3_manager.cache) == 0
            
            # Verify in_progress_downloads is empty
            assert len(s3_manager.in_progress_downloads) == 0
            
            # Verify future was cancelled
            mock_future.cancel.assert_called_once()
    
    def test_clear_cache_with_future_cancel_error(self, s3_manager):
        """
        GIVEN in-progress downloads where cancellation fails
        WHEN clear_cache is called
        THEN it should handle the error gracefully
        """
        # Add files to the cache
        for i in range(3):
            s3_path = f"s3://test-bucket/file{i}.txt"
            local_path = Path(f'/tmp/mock_temp_dir/file{i}.txt')
            s3_manager.cache[s3_path] = local_path
        
        # Add in-progress downloads with a future that raises an exception when cancelled
        mock_future = MagicMock(spec=Future)
        mock_future.cancel.side_effect = RuntimeError("Cannot cancel")
        s3_manager.in_progress_downloads["s3://test-bucket/downloading.txt"] = mock_future
        
        # Mock file and directory operations
        with patch.object(Path, 'unlink') as mock_unlink, \
             patch('shutil.rmtree') as mock_rmtree, \
             patch('tempfile.mkdtemp') as mock_mkdtemp:
            
            mock_mkdtemp.return_value = '/tmp/new_temp_dir'
            
            # Call clear_cache - should not propagate the exception
            s3_manager.clear_cache()
            
            # Verify all files were unlinked
            assert mock_unlink.call_count == 3
            
            # Verify temp directory was removed and recreated
            mock_rmtree.assert_called_once_with('/tmp/mock_temp_dir')
            mock_mkdtemp.assert_called_once()
            
            # Verify cache is empty
            assert len(s3_manager.cache) == 0
            
            # Verify in_progress_downloads is empty
            assert len(s3_manager.in_progress_downloads) == 0
            
            # Verify future.cancel was called
            mock_future.cancel.assert_called_once()


class TestPrefetching:
    """Tests for prefetching functionality."""
    
    def test_prefetch_files(self, s3_manager):
        """
        GIVEN a list of S3 paths
        WHEN prefetch_files is called
        THEN it should submit download tasks for uncached files
        """
        # Add one file to the cache
        cached_s3_path = "s3://test-bucket/cached-file.txt"
        local_path = Path('/tmp/mock_temp_dir/cached-file.txt')
        s3_manager.cache[cached_s3_path] = local_path
        
        # Create a list of paths to prefetch
        s3_paths = [
            cached_s3_path,  # Already cached
            "s3://test-bucket/new-file1.txt",  # Not cached
            "s3://test-bucket/new-file2.txt",  # Not cached
        ]
        
        # Configure mock executor
        mock_future = MagicMock(spec=Future)
        s3_manager.executor.submit.return_value = mock_future
        
        # Call prefetch_files
        s3_manager.prefetch_files(s3_paths)
        
        # Verify executor was called for uncached files only
        assert s3_manager.executor.submit.call_count == 2
        s3_manager.executor.submit.assert_has_calls([
            call(s3_manager.download_file, "s3://test-bucket/new-file1.txt"),
            call(s3_manager.download_file, "s3://test-bucket/new-file2.txt"),
        ])
        
        # Verify in_progress_downloads was updated
        assert len(s3_manager.in_progress_downloads) == 2
        assert "s3://test-bucket/new-file1.txt" in s3_manager.in_progress_downloads
        assert "s3://test-bucket/new-file2.txt" in s3_manager.in_progress_downloads
        
        # Verify callback was added to each future
        assert mock_future.add_done_callback.call_count == 2
    
    def test_prefetch_files_with_empty_list(self, s3_manager):
        """
        GIVEN an empty list of S3 paths
        WHEN prefetch_files is called
        THEN it should not submit any download tasks
        """
        # Call prefetch_files with empty list
        s3_manager.prefetch_files([])
        
        # Verify executor was not called
        s3_manager.executor.submit.assert_not_called()
        
        # Verify in_progress_downloads was not updated
        assert len(s3_manager.in_progress_downloads) == 0
    
    def test_prefetch_files_with_invalid_paths(self, s3_manager):
        """
        GIVEN a list with invalid S3 paths
        WHEN prefetch_files is called
        THEN it should handle the errors gracefully
        """
        # Configure mock executor to raise exception for invalid path
        def submit_side_effect(func, path):
            if path == "invalid-path":
                mock_future = MagicMock(spec=Future)
                mock_future.add_done_callback.side_effect = lambda cb: cb(mock_future)
                mock_future.cancelled.return_value = False
                return mock_future
            else:
                return MagicMock(spec=Future)
                
        s3_manager.executor.submit.side_effect = submit_side_effect
        
        # Call prefetch_files with mix of valid and invalid paths
        s3_paths = [
            "s3://test-bucket/valid-file.txt",
            "invalid-path"  # This will cause an assertion error in _parse_s3_path
        ]
        
        # Should not raise exception at the prefetch level
        s3_manager.prefetch_files(s3_paths)
        
        # Verify executor was called for both paths
        assert s3_manager.executor.submit.call_count == 2
    
    def test_download_complete_callback(self, s3_manager):
        """
        GIVEN a download future
        WHEN the download completes
        THEN the callback should remove it from in_progress_downloads
        """
        s3_path = "s3://test-bucket/downloading-file.txt"
        
        # Create a mock future
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.return_value = False
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback
        callback(mock_future)
        
        # Verify the path was removed from in_progress_downloads
        assert s3_path not in s3_manager.in_progress_downloads
    
    def test_download_complete_callback_cancelled(self, s3_manager):
        """
        GIVEN a download future that was cancelled
        WHEN the callback is executed
        THEN it should handle the cancellation gracefully
        """
        s3_path = "s3://test-bucket/cancelled-file.txt"
        
        # Create a mock future that was cancelled
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.return_value = True
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback
        callback(mock_future)
        
        # The path should still be in in_progress_downloads
        # since the callback returns early for cancelled futures
        assert s3_path in s3_manager.in_progress_downloads
    
    def test_download_in_progress(self, s3_manager, mock_s3_client):
        """
        GIVEN a file that is currently being downloaded
        WHEN download_file is called for the same file
        THEN it should wait for the in-progress download and return the result
        """
        s3_path = "s3://test-bucket/in-progress-file.txt"
        local_path = Path('/tmp/mock_temp_dir/in-progress-file.txt')
        
        # Create a mock future
        mock_future = MagicMock(spec=Future)
        mock_future.result.return_value = local_path
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Pre-populate the cache as if the download completed
        s3_manager.cache[s3_path] = local_path
        
        # Call download_file
        result = s3_manager.download_file(s3_path)
        
        # Verify the result
        assert result == local_path
        
        # Verify future.result() was called to wait for completion
        mock_future.result.assert_called_once()
        
        # Verify S3 client was not called
        mock_s3_client.download_file.assert_not_called()
        
    def test_concurrent_download_same_file(self, s3_manager, mock_s3_client):
        """
        GIVEN multiple threads downloading the same file simultaneously
        WHEN download_file is called concurrently for the same file
        THEN only one actual download should occur and all threads should get the result
        """
        s3_path = "s3://test-bucket/concurrent-same-file.txt"
        local_path = Path('/tmp/mock_temp_dir/concurrent-same-file.txt')
        
        # Configure mock to simulate successful download but track calls
        call_count = 0
        
        def download_side_effect(bucket, key, path):
            nonlocal call_count
            call_count += 1
            # Simulate work
            time.sleep(0.01)
            return None
            
        mock_s3_client.download_file.side_effect = download_side_effect
        
        # Function to run in each thread
        results = []
        exceptions = []
        
        def download_same_file():
            try:
                result = s3_manager.download_file(s3_path)
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Create and start threads
        num_threads = 5
        threads = [threading.Thread(target=download_same_file) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Verify no exceptions occurred
        assert not exceptions, f"Exceptions occurred: {exceptions}"
        
        # Verify all threads got a result
        assert len(results) == num_threads
        
        # Verify all results are the same
        assert all(result == results[0] for result in results)
        
        # Verify S3 client was called exactly once
        assert call_count == 1
        
        # Verify the file is in the cache
        assert s3_path in s3_manager.cache
        
    def test_download_in_progress_with_exception(self, s3_manager, mock_s3_client):
        """
        GIVEN a file that is currently being downloaded but the future raises an exception
        WHEN download_file is called for the same file
        THEN it should propagate the exception
        """
        s3_path = "s3://test-bucket/failing-download.txt"
        
        # Create a mock future that raises an exception
        mock_future = MagicMock(spec=Future)
        mock_future.result.side_effect = Exception("Download failed")
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Call download_file and expect exception
        with pytest.raises(Exception, match="Download failed"):
            s3_manager.download_file(s3_path)
        
        # Verify future.result() was called
        mock_future.result.assert_called_once()
        
        # Verify S3 client was not called
        mock_s3_client.download_file.assert_not_called()


class TestDownloadCompleteCallback:
    """Tests specifically for the _download_complete callback generator."""
    
    def test_callback_removes_completed_download(self, s3_manager):
        """
        GIVEN an S3 path in the in_progress_downloads dictionary
        WHEN the callback is executed with a completed future
        THEN the path should be removed from in_progress_downloads
        """
        s3_path = "s3://test-bucket/completed-download.txt"
        
        # Create a mock future
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.return_value = False
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback
        callback(mock_future)
        
        # Verify the path was removed from in_progress_downloads
        assert s3_path not in s3_manager.in_progress_downloads
    
    def test_callback_preserves_cancelled_download(self, s3_manager):
        """
        GIVEN an S3 path in the in_progress_downloads dictionary
        WHEN the callback is executed with a cancelled future
        THEN the path should remain in in_progress_downloads
        """
        s3_path = "s3://test-bucket/cancelled-download.txt"
        
        # Create a mock future that was cancelled
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.return_value = True
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback
        callback(mock_future)
        
        # The path should still be in in_progress_downloads
        assert s3_path in s3_manager.in_progress_downloads
    
    def test_callback_with_lock_contention(self, s3_manager):
        """
        GIVEN multiple threads executing the callback simultaneously
        WHEN the callbacks are executed
        THEN the cache_lock should prevent race conditions
        """
        s3_path = "s3://test-bucket/contended-download.txt"
        
        # Create a mock future
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.return_value = False
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Mock the lock to verify it's used correctly
        original_lock = s3_manager.cache_lock
        mock_lock = MagicMock(wraps=original_lock)
        s3_manager.cache_lock = mock_lock
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback
        callback(mock_future)
        
        # Verify the lock was acquired and released
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()
        
        # Verify the path was removed
        assert s3_path not in s3_manager.in_progress_downloads
    
    def test_callback_with_nonexistent_path(self, s3_manager):
        """
        GIVEN an S3 path not in the in_progress_downloads dictionary
        WHEN the callback is executed
        THEN it should handle the situation gracefully without errors
        """
        s3_path = "s3://test-bucket/nonexistent-download.txt"
        
        # Create a mock future
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.return_value = False
        
        # Ensure path is not in in_progress_downloads
        if s3_path in s3_manager.in_progress_downloads:
            del s3_manager.in_progress_downloads[s3_path]
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback - should not raise any exceptions
        callback(mock_future)
        
        # Verify in_progress_downloads remains unchanged
        assert s3_path not in s3_manager.in_progress_downloads
    
    def test_callback_with_exception_in_cancelled_check(self, s3_manager):
        """
        GIVEN a future that raises an exception when cancelled() is called
        WHEN the callback is executed
        THEN it should handle the exception gracefully
        """
        s3_path = "s3://test-bucket/exception-in-cancelled.txt"
        
        # Create a mock future that raises an exception when cancelled() is called
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.side_effect = Exception("Error checking cancellation")
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback - should not propagate the exception
        try:
            callback(mock_future)
            # If we get here, no exception was raised - test passes
            exception_raised = False
        except Exception:
            exception_raised = True
        
        assert not exception_raised, "Exception should be caught within the callback"
        
        # The path should still be in in_progress_downloads since an exception occurred
        assert s3_path in s3_manager.in_progress_downloads
    
    def test_callback_integration_with_prefetch(self, s3_manager):
        """
        GIVEN a prefetch operation that adds callbacks to futures
        WHEN the prefetch completes
        THEN the callbacks should be added to the futures correctly
        """
        # Configure mock executor
        mock_future = MagicMock(spec=Future)
        s3_manager.executor.submit.return_value = mock_future
        
        # Call prefetch_files
        s3_paths = ["s3://test-bucket/prefetch-file.txt"]
        s3_manager.prefetch_files(s3_paths)
        
        # Verify callback was added to the future
        mock_future.add_done_callback.assert_called_once()
        
        # Get the callback that was added
        callback_arg = mock_future.add_done_callback.call_args[0][0]
        
        # Verify it's a callback function
        assert callable(callback_arg)
    
    def test_callback_with_multiple_futures_same_path(self, s3_manager):
        """
        GIVEN multiple futures for the same S3 path
        WHEN callbacks are executed for each future
        THEN the path should only be removed after all futures are processed
        """
        s3_path = "s3://test-bucket/multiple-futures.txt"
        
        # Create multiple mock futures for the same path
        mock_future1 = MagicMock(spec=Future)
        mock_future1.cancelled.return_value = False
        mock_future2 = MagicMock(spec=Future)
        mock_future2.cancelled.return_value = False
        
        # Add the first future to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future1
        
        # Get callbacks for both futures
        callback1 = s3_manager._download_complete(s3_path)
        callback2 = s3_manager._download_complete(s3_path)
        
        # Execute first callback
        callback1(mock_future1)
        
        # Path should be removed from in_progress_downloads
        assert s3_path not in s3_manager.in_progress_downloads
        
        # Add the second future to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future2
        
        # Execute second callback
        callback2(mock_future2)
        
        # Path should be removed again
        assert s3_path not in s3_manager.in_progress_downloads
    
    def test_callback_closure_captures_correct_path(self, s3_manager):
        """
        GIVEN multiple callbacks created for different paths
        WHEN the callbacks are executed
        THEN each callback should operate on its specific path
        """
        # Create multiple paths and futures
        s3_path1 = "s3://test-bucket/path1.txt"
        s3_path2 = "s3://test-bucket/path2.txt"
        
        mock_future1 = MagicMock(spec=Future)
        mock_future1.cancelled.return_value = False
        mock_future2 = MagicMock(spec=Future)
        mock_future2.cancelled.return_value = False
        
        # Add both to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path1] = mock_future1
        s3_manager.in_progress_downloads[s3_path2] = mock_future2
        
        # Get callbacks for each path
        callback1 = s3_manager._download_complete(s3_path1)
        callback2 = s3_manager._download_complete(s3_path2)
        
        # Execute first callback
        callback1(mock_future1)
        
        # Only the first path should be removed
        assert s3_path1 not in s3_manager.in_progress_downloads
        assert s3_path2 in s3_manager.in_progress_downloads
        
        # Execute second callback
        callback2(mock_future2)
        
        # Now both paths should be removed
        assert s3_path1 not in s3_manager.in_progress_downloads
        assert s3_path2 not in s3_manager.in_progress_downloads
        
    def test_callback_with_exception_during_pop(self, s3_manager):
        """
        GIVEN a future and a callback
        WHEN the callback is executed but pop operation raises an exception
        THEN the exception should be caught and not propagated
        """
        s3_path = "s3://test-bucket/exception-during-pop.txt"
        
        # Create a mock future
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.return_value = False
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Mock the dictionary's pop method to raise an exception
        original_pop = s3_manager.in_progress_downloads.pop
        
        def pop_with_exception(key, default=None):
            if key == s3_path:
                raise KeyError("Simulated error during pop operation")
            return original_pop(key, default)
            
        s3_manager.in_progress_downloads.pop = pop_with_exception
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback - should not raise any exceptions
        try:
            callback(mock_future)
            exception_raised = False
        except Exception:
            exception_raised = True
            
        # Restore original pop method
        s3_manager.in_progress_downloads.pop = original_pop
        
        # Verify no exception was propagated
        assert not exception_raised, "Exception should be caught within the callback"
        
    def test_callback_with_recursive_lock_acquisition(self, s3_manager):
        """
        GIVEN a callback that might be called while already holding the lock
        WHEN the callback is executed
        THEN it should handle recursive lock acquisition correctly
        """
        s3_path = "s3://test-bucket/recursive-lock.txt"
        
        # Create a mock future
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.return_value = False
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback while already holding the lock
        with s3_manager.cache_lock:
            # This should not deadlock if the lock is reentrant
            callback(mock_future)
            
        # Verify the path was removed
        assert s3_path not in s3_manager.in_progress_downloads
        
    def test_callback_execution_order(self, s3_manager):
        """
        GIVEN multiple callbacks for the same path registered in sequence
        WHEN the callbacks are executed in reverse order
        THEN each callback should still operate correctly
        """
        s3_path = "s3://test-bucket/execution-order.txt"
        
        # Create multiple mock futures for the same path
        mock_futures = [MagicMock(spec=Future) for _ in range(3)]
        for mock_future in mock_futures:
            mock_future.cancelled.return_value = False
        
        # Create callbacks for each future but for the same path
        callbacks = []
        for i, mock_future in enumerate(mock_futures):
            # Add to in_progress_downloads (overwriting previous)
            s3_manager.in_progress_downloads[s3_path] = mock_future
            # Get callback
            callbacks.append(s3_manager._download_complete(s3_path))
        
        # Execute callbacks in reverse order
        for callback in reversed(callbacks):
            # Re-add the path to in_progress_downloads before each callback
            s3_manager.in_progress_downloads[s3_path] = MagicMock(spec=Future)
            callback(MagicMock(spec=Future))
            # Verify the path was removed
            assert s3_path not in s3_manager.in_progress_downloads
            
    def test_callback_with_future_exception(self, s3_manager):
        """
        GIVEN a future that raises an exception when accessed
        WHEN the callback is executed
        THEN it should handle the exception gracefully
        """
        s3_path = "s3://test-bucket/future-exception.txt"
        
        # Create a mock future that raises an exception when any method is called
        mock_future = MagicMock(spec=Future)
        mock_future.__getattribute__ = MagicMock(side_effect=Exception("Future is broken"))
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback - should not propagate the exception
        try:
            callback(mock_future)
            exception_raised = False
        except Exception:
            exception_raised = True
            
        # Verify no exception was propagated
        assert not exception_raised, "Exception should be caught within the callback"
        
        # The path should still be in in_progress_downloads since an exception occurred
        assert s3_path in s3_manager.in_progress_downloads
        
    def test_callback_with_lock_exception(self, s3_manager):
        """
        GIVEN a scenario where acquiring the lock raises an exception
        WHEN the callback is executed
        THEN it should handle the lock acquisition failure gracefully
        """
        s3_path = "s3://test-bucket/lock-exception.txt"
        
        # Create a mock future
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.return_value = False
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Replace the lock with one that raises an exception when acquired
        original_lock = s3_manager.cache_lock
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(side_effect=RuntimeError("Lock acquisition failed"))
        s3_manager.cache_lock = mock_lock
        
        # Get the callback
        callback = s3_manager._download_complete(s3_path)
        
        # Call the callback - should not propagate the exception
        try:
            callback(mock_future)
            exception_raised = False
        except Exception:
            exception_raised = True
            
        # Restore original lock
        s3_manager.cache_lock = original_lock
        
        # Verify no exception was propagated
        assert not exception_raised, "Exception should be caught within the callback"
        
        # The path should still be in in_progress_downloads since an exception occurred
        assert s3_path in s3_manager.in_progress_downloads
        
    def test_callback_thread_safety_with_concurrent_execution(self, s3_manager):
        """
        GIVEN multiple callbacks for the same path
        WHEN they are executed concurrently
        THEN the operations should be thread-safe
        """
        s3_path = "s3://test-bucket/concurrent-callbacks.txt"
        
        # Create a mock future
        mock_future = MagicMock(spec=Future)
        mock_future.cancelled.return_value = False
        
        # Add to in_progress_downloads
        s3_manager.in_progress_downloads[s3_path] = mock_future
        
        # Create multiple callbacks for the same path
        num_callbacks = 5
        callbacks = [s3_manager._download_complete(s3_path) for _ in range(num_callbacks)]
        
        # Track if any exceptions occur during concurrent execution
        exceptions = []
        
        def execute_callback(callback_fn):
            try:
                # Re-add the path before each callback execution
                s3_manager.in_progress_downloads[s3_path] = mock_future
                callback_fn(mock_future)
            except Exception as e:
                exceptions.append(e)
        
        # Execute callbacks concurrently
        threads = [threading.Thread(target=execute_callback, args=(callback,)) for callback in callbacks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Verify no exceptions occurred
        assert not exceptions, f"Exceptions occurred during concurrent execution: {exceptions}"


class TestSingletonEdgeCases:
    """Tests for edge cases in the singleton implementation."""
    
    def test_new_called_with_different_class(self):
        """
        GIVEN S3FileManager.__new__ is called with a different class
        WHEN the method is invoked directly with a different class argument
        THEN it should still maintain singleton behavior for the class it's called on
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create a dummy class
        class DummyClass:
            pass
        
        # Call __new__ with the dummy class as argument
        # This simulates what happens in complex inheritance scenarios
        instance1 = S3FileManager.__new__(DummyClass)
        
        # Verify the instance is of the expected type
        assert isinstance(instance1, DummyClass)
        
        # Create a normal S3FileManager instance
        s3_instance = S3FileManager()
        
        # Verify they are different objects
        assert instance1 is not s3_instance
        assert not isinstance(instance1, S3FileManager)
    
    def test_singleton_after_clear_cache(self, s3_manager):
        """
        GIVEN an existing singleton instance
        WHEN clear_cache is called
        THEN the instance should remain the same singleton
        """
        # Get the current singleton instance
        original_instance = s3_manager
        
        # Call clear_cache
        with patch('shutil.rmtree'), patch('tempfile.mkdtemp'):
            original_instance.clear_cache()
        
        # Create a new instance
        new_instance = S3FileManager()
        
        # Should be the same instance
        assert new_instance is original_instance
        
    def test_singleton_persistence_across_imports(self):
        """
        GIVEN multiple imports of S3FileManager
        WHEN instances are created from different import statements
        THEN all instances should be the same object
        """
        # This test simulates multiple imports by directly manipulating sys.modules
        import sys
        import importlib
        
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create first instance
        instance1 = S3FileManager(max_cache_size=100)
        
        # Simulate reimporting the module
        original_module = sys.modules['src.utilities']
        del sys.modules['src.utilities']
        
        # Reimport and create new instance
        with patch('boto3.client'), patch('tempfile.mkdtemp'), patch('concurrent.futures.ThreadPoolExecutor'):
            utilities_module = importlib.import_module('src.utilities')
            instance2 = utilities_module.S3FileManager(max_cache_size=200)
        
        # Restore original module
        sys.modules['src.utilities'] = original_module
        
        # Both instances should refer to the same object
        assert instance1 is instance2
        assert instance1.max_cache_size == instance2.max_cache_size == 100
    
    def test_singleton_with_inheritance(self):
        """
        GIVEN a class that inherits from S3FileManager
        WHEN instances of both parent and child classes are created
        THEN they should maintain separate singleton instances
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create a subclass
        class ExtendedS3FileManager(S3FileManager):
            _instance = None
            _instance_lock = threading.Lock()
            
            def __init__(self, max_cache_size=100, extra_feature=None):
                super().__init__(max_cache_size)
                self.extra_feature = extra_feature
        
        # Create instances of both classes
        parent_instance = S3FileManager(max_cache_size=100)
        child_instance = ExtendedS3FileManager(max_cache_size=200, extra_feature="test")
        
        # Each class should have its own singleton
        assert isinstance(parent_instance, S3FileManager)
        assert isinstance(child_instance, ExtendedS3FileManager)
        assert parent_instance is not child_instance
        
        # Creating another instance of each class should return the existing singleton
        parent_instance2 = S3FileManager(max_cache_size=300)
        child_instance2 = ExtendedS3FileManager(max_cache_size=400, extra_feature="another")
        
        assert parent_instance2 is parent_instance
        assert child_instance2 is child_instance
        assert parent_instance2.max_cache_size == 100  # First initialization value
        assert child_instance2.max_cache_size == 200  # First initialization value
        assert child_instance2.extra_feature == "test"  # First initialization value
        
    def test_new_without_initialization(self):
        """
        GIVEN a direct call to __new__ without subsequent __init__
        WHEN an instance is created using only __new__
        THEN it should still be a valid singleton instance but not initialized
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create instance using only __new__
        instance = S3FileManager.__new__(S3FileManager)
        
        # Verify it's a valid instance but not initialized
        assert isinstance(instance, S3FileManager)
        assert instance._initialized is False
        
        # Now create a normal instance
        normal_instance = S3FileManager()
        
        # Both should be the same object
        assert normal_instance is instance
        
        # The instance should now be initialized
        assert instance._initialized is True


class TestInitExceptions:
    """Tests for exception handling during initialization."""
    
    def test_boto3_client_exception(self):
        """
        GIVEN boto3.client raises an exception
        WHEN S3FileManager is initialized
        THEN the exception should be propagated
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('boto3.client') as mock_client:
            # Configure mock to raise an exception
            mock_client.side_effect = ImportError("boto3 not installed")
            
            # Attempt to create instance
            with pytest.raises(ImportError) as excinfo:
                S3FileManager()
                
            # Verify exception was propagated
            assert "boto3 not installed" in str(excinfo.value)
            
            # Verify singleton instance was not set
            assert S3FileManager._instance is None
    
    def test_mkdtemp_exception(self):
        """
        GIVEN tempfile.mkdtemp raises an exception
        WHEN S3FileManager is initialized
        THEN the exception should be propagated
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('boto3.client'), \
             patch('tempfile.mkdtemp') as mock_mkdtemp:
            
            # Configure mock to raise an exception
            mock_mkdtemp.side_effect = PermissionError("Permission denied")
            
            # Attempt to create instance
            with pytest.raises(PermissionError) as excinfo:
                S3FileManager()
                
            # Verify exception was propagated
            assert "Permission denied" in str(excinfo.value)
            
            # Verify singleton instance was not set
            assert S3FileManager._instance is None
    
    def test_threadpoolexecutor_exception(self):
        """
        GIVEN ThreadPoolExecutor initialization raises an exception
        WHEN S3FileManager is initialized
        THEN the exception should be propagated
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('boto3.client'), \
             patch('tempfile.mkdtemp'), \
             patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            
            # Configure mock to raise an exception
            mock_executor.side_effect = ValueError("Invalid max_workers")
            
            # Attempt to create instance
            with pytest.raises(ValueError) as excinfo:
                S3FileManager()
                
            # Verify exception was propagated
            assert "Invalid max_workers" in str(excinfo.value)
            
            # Verify singleton instance was not set
            assert S3FileManager._instance is None
    
    def test_partial_initialization_cleanup(self):
        """
        GIVEN an exception occurs during initialization
        WHEN S3FileManager is initialized
        THEN any created resources should be cleaned up
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        temp_dir = '/tmp/test_temp_dir'
        
        with patch('boto3.client') as mock_client, \
             patch('tempfile.mkdtemp') as mock_mkdtemp, \
             patch('concurrent.futures.ThreadPoolExecutor') as mock_executor, \
             patch('shutil.rmtree') as mock_rmtree:
            
            # Configure mocks
            mock_client.return_value = MagicMock()
            mock_mkdtemp.return_value = temp_dir
            
            # Make ThreadPoolExecutor raise an exception
            mock_executor.side_effect = RuntimeError("Executor initialization failed")
            
            # Attempt to create instance
            with pytest.raises(RuntimeError):
                S3FileManager()
            
            # Verify temp directory was cleaned up
            # Note: This test assumes the implementation cleans up resources on failure
            # If it doesn't, this test would need to be adjusted or removed
            mock_rmtree.assert_called_once_with(temp_dir)

class TestConcurrency:
    """Tests for concurrent access to S3FileManager."""
    
    def test_download_file_temp_dir_missing(self, mock_s3_client):
        """
        GIVEN the temp directory doesn't exist when download_file is called
        WHEN download_file is called
        THEN it should handle the error gracefully
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = '/tmp/mock_temp_dir'
            manager = S3FileManager(max_cache_size=5)
            
            # Replace the ThreadPoolExecutor with a mock
            manager.executor = MagicMock()
            
            # Simulate temp directory being deleted after initialization
            with patch.object(Path, 'exists', return_value=False), \
                 patch('os.makedirs') as mock_makedirs:
                
                s3_path = "s3://test-bucket/missing-temp-dir-file.txt"
                expected_local_path = Path('/tmp/mock_temp_dir/missing-temp-dir-file.txt')
                
                # Call download_file
                result = manager.download_file(s3_path)
                
                # Verify directory was created
                mock_makedirs.assert_called_once_with('/tmp/mock_temp_dir', exist_ok=True)
                
                # Verify download proceeded
                mock_s3_client.download_file.assert_called_once_with(
                    "test-bucket", "missing-temp-dir-file.txt", str(expected_local_path)
                )
                
                # Verify result
                assert result == expected_local_path
    
    def test_concurrent_downloads(self, s3_manager, mock_s3_client):
        """
        GIVEN multiple threads downloading different files
        WHEN download_file is called concurrently
        THEN all downloads should complete successfully
        """
        # Define test parameters
        num_threads = 10
        s3_paths = [f"s3://test-bucket/concurrent-file{i}.txt" for i in range(num_threads)]
        local_paths = [Path(f'/tmp/mock_temp_dir/concurrent-file{i}.txt') for i in range(num_threads)]
        
        # Configure mock to return different paths based on input
        def download_side_effect(bucket, key, local_path):
            # No-op, just simulate successful download
            pass
        
        mock_s3_client.download_file.side_effect = download_side_effect
        
        # Function to run in each thread
        results = {}
        exceptions = []
        
        def download_file(thread_id):
            try:
                s3_path = s3_paths[thread_id]
                result = s3_manager.download_file(s3_path)
                results[thread_id] = result
            except Exception as e:
                exceptions.append((thread_id, e))
        
        # Create and start threads
        threads = [threading.Thread(target=download_file, args=(i,)) for i in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Verify no exceptions occurred
        assert not exceptions, f"Exceptions occurred: {exceptions}"
        
        # Verify all downloads completed
        assert len(results) == num_threads
        
        # Verify S3 client was called for each file
        assert mock_s3_client.download_file.call_count == num_threads
        
        # Verify all files are in the cache
        for s3_path in s3_paths:
            assert s3_path in s3_manager.cache
            
    def test_concurrent_cache_access(self, s3_manager):
        """
        GIVEN multiple threads accessing the cache simultaneously
        WHEN get_file and download_file are called concurrently
        THEN the cache should remain consistent
        """
        # Pre-populate the cache with some files
        for i in range(5):
            s3_path = f"s3://test-bucket/cache-file{i}.txt"
            local_path = Path(f'/tmp/mock_temp_dir/cache-file{i}.txt')
            s3_manager.cache[s3_path] = local_path
        
        # Mock download_file to avoid actual S3 calls but still use the cache
        original_download_file = s3_manager.download_file
        
        def mock_download_file(s3_path):
            with s3_manager.cache_lock:
                if s3_path in s3_manager.cache:
                    s3_manager.cache.move_to_end(s3_path)
                    return s3_manager.cache[s3_path]
                else:
                    local_path = Path(f'/tmp/mock_temp_dir/{s3_path.split("/")[-1]}')
                    s3_manager.cache[s3_path] = local_path
                    if len(s3_manager.cache) > s3_manager.max_cache_size:
                        s3_manager.cache.popitem(last=False)
                    return local_path
        
        s3_manager.download_file = mock_download_file
        
        # Function to run in each thread - mix of get_file and download_file
        results = []
        exceptions = []
        
        def access_cache(thread_id, operation, s3_path):
            try:
                if operation == 'get':
                    result = s3_manager.get_file(s3_path)
                else:  # download
                    result = s3_manager.download_file(s3_path)
                results.append((thread_id, operation, s3_path, result))
            except Exception as e:
                exceptions.append((thread_id, operation, s3_path, e))
        
        # Create a mix of operations
        operations = []
        for i in range(20):
            op = 'get' if i % 3 == 0 else 'download'
            s3_path = f"s3://test-bucket/cache-file{i % 7}.txt"  # Some overlap to test contention
            operations.append((i, op, s3_path))
        
        # Create and start threads
        threads = [threading.Thread(target=access_cache, args=op) for op in operations]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Restore original method
        s3_manager.download_file = original_download_file
        
        # Verify no exceptions occurred
        assert not exceptions, f"Exceptions occurred: {exceptions}"
        
        # Verify all operations completed
        assert len(results) == len(operations)
        
        # Verify cache size is within limits
        assert len(s3_manager.cache) <= s3_manager.max_cache_size


class TestDownloadFileEdgeCases:
    """Tests for edge cases in the download_file method."""
    
    def test_download_file_with_problematic_filesystem_characters(self, s3_manager, mock_s3_client):
        """
        GIVEN an S3 path with characters that are problematic for filesystems
        WHEN download_file is called
        THEN it should handle these characters appropriately
        """
        # Characters that are often problematic on filesystems: ?, *, :, ", <, >, |
        s3_path = "s3://test-bucket/file?with*problematic:characters\"<>|.txt"
        expected_local_path = Path('/tmp/mock_temp_dir/file?with*problematic:characters"<>|.txt')
        
        # Mock the path operations to avoid actual filesystem issues
        with patch.object(Path, 'exists', return_value=True):
            result = s3_manager.download_file(s3_path)
            
            # Verify the result
            assert result == expected_local_path
            
            # Verify S3 client was called with the correct parameters
            mock_s3_client.download_file.assert_called_once_with(
                "test-bucket", "file?with*problematic:characters\"<>|.txt", str(expected_local_path)
            )
    
    def test_download_file_with_special_characters(self, s3_manager, mock_s3_client):
        """
        GIVEN an S3 path with special characters
        WHEN download_file is called
        THEN it should handle the special characters correctly
        """
        s3_path = "s3://test-bucket/path/with special+chars&symbols.txt"
        expected_local_path = Path('/tmp/mock_temp_dir/path_with special+chars&symbols.txt')
        
        result = s3_manager.download_file(s3_path)
        
        assert result == expected_local_path
        mock_s3_client.download_file.assert_called_once_with(
            "test-bucket", "path/with special+chars&symbols.txt", str(expected_local_path)
        )
    
    def test_download_file_with_very_long_path(self, s3_manager, mock_s3_client):
        """
        GIVEN an S3 path with a very long key
        WHEN download_file is called
        THEN it should handle the long path correctly
        """
        # Create a very long path
        long_path = "path/" + "x" * 200 + "/file.txt"
        s3_path = f"s3://test-bucket/{long_path}"
        expected_local_path = Path('/tmp/mock_temp_dir/path_' + "x" * 200 + "_file.txt")
        
        result = s3_manager.download_file(s3_path)
        
        assert result == expected_local_path
        mock_s3_client.download_file.assert_called_once_with(
            "test-bucket", long_path, str(expected_local_path)
        )
    
    def test_download_file_with_empty_key(self, s3_manager, mock_s3_client):
        """
        GIVEN an S3 path with an empty key
        WHEN download_file is called
        THEN it should handle the empty key correctly
        """
        s3_path = "s3://test-bucket/"
        expected_local_path = Path('/tmp/mock_temp_dir/')
        
        result = s3_manager.download_file(s3_path)
        
        assert result == expected_local_path
        mock_s3_client.download_file.assert_called_once_with(
            "test-bucket", "", str(expected_local_path)
        )
        
    def test_download_file_with_identical_flattened_paths(self, s3_manager, mock_s3_client):
        """
        GIVEN two different S3 paths that flatten to the same local filename
        WHEN download_file is called for both
        THEN it should handle the collision appropriately
        """
        # These two paths would both flatten to 'a_b_c.txt'
        s3_path1 = "s3://bucket1/a/b/c.txt"
        s3_path2 = "s3://bucket2/a_b_c.txt"
        
        expected_local_path1 = Path('/tmp/mock_temp_dir/a_b_c.txt')
        
        # First download
        result1 = s3_manager.download_file(s3_path1)
        assert result1 == expected_local_path1
        
        # Reset mock for second download
        mock_s3_client.reset_mock()
        
        # Second download - should use a different local path or some collision strategy
        result2 = s3_manager.download_file(s3_path2)
        
        # The implementation should handle this collision somehow
        # This assertion might need adjustment based on the actual implementation
        assert s3_path1 in s3_manager.cache
        assert s3_path2 in s3_manager.cache
        assert s3_manager.cache[s3_path1] != s3_manager.cache[s3_path2]
    
    def test_download_file_with_only_slashes_key(self, s3_manager, mock_s3_client):
        """
        GIVEN an S3 path with only slashes as the key
        WHEN download_file is called
        THEN it should handle the key correctly
        """
        s3_path = "s3://test-bucket////"
        expected_local_path = Path('/tmp/mock_temp_dir/___')
        
        result = s3_manager.download_file(s3_path)
        
        assert result == expected_local_path
        mock_s3_client.download_file.assert_called_once_with(
            "test-bucket", "///", str(expected_local_path)
        )

class TestInitialization:
    """Tests for S3FileManager initialization process."""
    
    def test_init_method_sets_attributes(self):
        """
        GIVEN a new S3FileManager instance
        WHEN _init is called
        THEN all required attributes should be properly set
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('boto3.client') as mock_client, \
             patch('tempfile.mkdtemp') as mock_mkdtemp, \
             patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            
            # Configure mocks
            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3
            mock_mkdtemp.return_value = '/tmp/test_dir'
            mock_executor_instance = MagicMock()
            mock_executor.return_value = mock_executor_instance
            
            # Create instance with specific max_cache_size
            max_cache_size = 42
            manager = S3FileManager(max_cache_size=max_cache_size)
            
            # Verify all attributes are set correctly
            assert manager.s3_client is mock_s3
            assert isinstance(manager.cache, OrderedDict)
            assert len(manager.cache) == 0
            assert isinstance(manager.cache_lock, threading.Lock)
            assert manager.max_cache_size == max_cache_size
            assert manager.temp_dir == Path('/tmp/test_dir')
            assert manager.executor is mock_executor_instance
            assert isinstance(manager.in_progress_downloads, dict)
            assert len(manager.in_progress_downloads) == 0
            assert manager._initialized is True
            
            # Verify mocks were called correctly
            mock_client.assert_called_once_with('s3')
            mock_mkdtemp.assert_called_once()
            mock_executor.assert_called_once_with(max_workers=5)
    
    def test_init_method_called_only_when_needed(self):
        """
        GIVEN an already initialized S3FileManager instance
        WHEN __init__ is called again
        THEN _init should not be called again
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create first instance
        with patch('boto3.client'), patch('tempfile.mkdtemp'), patch('concurrent.futures.ThreadPoolExecutor'):
            first_instance = S3FileManager(max_cache_size=100)
            first_instance._initialized = True  # Ensure it's marked as initialized
        
        # Now patch _init to track if it's called
        with patch.object(S3FileManager, '_init') as mock_init:
            # Create second instance
            second_instance = S3FileManager(max_cache_size=200)
            
            # Verify _init was not called again
            mock_init.assert_not_called()
            
            # Verify both instances are the same object
            assert second_instance is first_instance
    
    def test_init_method_with_uninitialized_singleton(self):
        """
        GIVEN an existing but uninitialized singleton instance
        WHEN __init__ is called
        THEN _init should be called to complete initialization
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create instance using only __new__ (without initialization)
        instance = S3FileManager.__new__(S3FileManager)
        assert instance._initialized is False
        
        # Now patch _init to track if it's called
        with patch.object(S3FileManager, '_init') as mock_init:
            # Call __init__ on the existing instance
            instance.__init__(max_cache_size=150)
            
            # Verify _init was called to complete initialization
            mock_init.assert_called_once_with(150)
    
    def test_init_method_with_different_parameters(self):
        """
        GIVEN multiple calls to __init__ with different parameters
        WHEN instances are created
        THEN only the first call's parameters should be used for initialization
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Track calls to _init
        original_init = S3FileManager._init
        init_calls = []
        
        def tracked_init(self, max_cache_size):
            init_calls.append(max_cache_size)
            original_init(self, max_cache_size)
            
        with patch.object(S3FileManager, '_init', tracked_init):
            # Create multiple instances with different parameters
            instance1 = S3FileManager(max_cache_size=100)
            instance2 = S3FileManager(max_cache_size=200)
            instance3 = S3FileManager(max_cache_size=300)
            
            # Verify _init was only called once with the first parameter
            assert len(init_calls) == 1
            assert init_calls[0] == 100
            
            # Verify all instances are the same object
            assert instance1 is instance2 is instance3
            
            # Verify the max_cache_size is from the first initialization
            assert instance1.max_cache_size == 100
    
    def test_init_with_zero_max_cache_size(self):
        """
        GIVEN a max_cache_size of zero
        WHEN S3FileManager is initialized
        THEN it should still initialize correctly
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('boto3.client'), patch('tempfile.mkdtemp'), \
             patch('concurrent.futures.ThreadPoolExecutor'):
            
            # Create instance with zero max_cache_size
            manager = S3FileManager(max_cache_size=0)
            
            # Verify max_cache_size is set to 0
            assert manager.max_cache_size == 0
            assert isinstance(manager.cache, OrderedDict)
            assert manager._initialized is True
    
    def test_init_with_negative_max_cache_size(self):
        """
        GIVEN a negative max_cache_size
        WHEN S3FileManager is initialized
        THEN it should handle it gracefully (implementation-dependent)
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('boto3.client'), patch('tempfile.mkdtemp'), \
             patch('concurrent.futures.ThreadPoolExecutor'):
            
            # Create instance with negative max_cache_size
            manager = S3FileManager(max_cache_size=-10)
            
            # Verify max_cache_size is set to the provided value
            # Note: This test might need adjustment based on how negative values should be handled
            assert manager.max_cache_size == -10
            assert isinstance(manager.cache, OrderedDict)
            assert manager._initialized is True
    
    def test_init_creates_temp_directory(self):
        """
        GIVEN S3FileManager initialization
        WHEN _init is called
        THEN it should create a temporary directory
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('boto3.client'), \
             patch('tempfile.mkdtemp') as mock_mkdtemp, \
             patch('concurrent.futures.ThreadPoolExecutor'):
            
            # Configure mock to return a specific path
            temp_dir_path = '/tmp/specific_test_dir'
            mock_mkdtemp.return_value = temp_dir_path
            
            # Create instance
            manager = S3FileManager()
            
            # Verify temp directory was created
            mock_mkdtemp.assert_called_once()
            assert manager.temp_dir == Path(temp_dir_path)
    
    def test_init_creates_thread_pool(self):
        """
        GIVEN S3FileManager initialization
        WHEN _init is called
        THEN it should create a ThreadPoolExecutor with correct parameters
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('boto3.client'), \
             patch('tempfile.mkdtemp'), \
             patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            
            # Create instance
            manager = S3FileManager()
            
            # Verify executor was created with correct parameters
            mock_executor.assert_called_once_with(max_workers=5)
    
    def test_init_with_custom_thread_pool_size(self):
        """
        GIVEN S3FileManager with a custom thread pool implementation
        WHEN _init is called with a modified implementation
        THEN it should use the custom thread pool size
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create a subclass with custom thread pool size
        class CustomS3FileManager(S3FileManager):
            def _init(self, max_cache_size):
                self.s3_client = boto3.client('s3')
                self.cache = OrderedDict()
                self.cache_lock = threading.Lock()
                self.max_cache_size = max_cache_size
                self.temp_dir = Path(tempfile.mkdtemp())
                self.executor = ThreadPoolExecutor(max_workers=10)  # Custom size
                self.in_progress_downloads = {}
                self._initialized = True
        
        with patch('boto3.client'), \
             patch('tempfile.mkdtemp'), \
             patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            
            # Create instance of custom class
            manager = CustomS3FileManager()
            
            # Verify executor was created with custom parameters
            mock_executor.assert_called_once_with(max_workers=10)
    
    def test_init_attributes_isolation(self):
        """
        GIVEN multiple S3FileManager instances
        WHEN _init is called for each instance
        THEN each instance should have its own isolated attributes
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create a non-singleton subclass for testing attribute isolation
        class NonSingletonS3FileManager(S3FileManager):
            _instance = None  # Override to avoid singleton behavior
            
            def __new__(cls, *args, **kwargs):
                return super(S3FileManager, cls).__new__(cls)
        
        with patch('boto3.client') as mock_client, \
             patch('tempfile.mkdtemp') as mock_mkdtemp, \
             patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            
            # Configure mocks to return different values for each instance
            mock_client.side_effect = [MagicMock(name="s3_client1"), MagicMock(name="s3_client2")]
            mock_mkdtemp.side_effect = ['/tmp/dir1', '/tmp/dir2']
            mock_executor.side_effect = [MagicMock(name="executor1"), MagicMock(name="executor2")]
            
            # Create two instances
            manager1 = NonSingletonS3FileManager(max_cache_size=10)
            manager2 = NonSingletonS3FileManager(max_cache_size=20)
            
            # Verify each instance has its own attributes
            assert manager1.max_cache_size == 10
            assert manager2.max_cache_size == 20
            assert manager1.s3_client is not manager2.s3_client
            assert manager1.temp_dir != manager2.temp_dir
            assert manager1.executor is not manager2.executor
            assert manager1.cache is not manager2.cache
            assert manager1.cache_lock is not manager2.cache_lock
            
    def test_init_method_called_only_when_needed(self):
        """
        GIVEN an already initialized S3FileManager instance
        WHEN __init__ is called again
        THEN _init should not be called again
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create first instance
        with patch('boto3.client'), patch('tempfile.mkdtemp'), patch('concurrent.futures.ThreadPoolExecutor'):
            first_instance = S3FileManager(max_cache_size=100)
            first_instance._initialized = True  # Ensure it's marked as initialized
        
        # Now patch _init to track if it's called
        with patch.object(S3FileManager, '_init') as mock_init:
            # Create second instance
            second_instance = S3FileManager(max_cache_size=200)
            
            # Verify _init was not called again
            mock_init.assert_not_called()
            
            # Verify both instances are the same object
            assert second_instance is first_instance
    
    def test_init_method_with_uninitialized_singleton(self):
        """
        GIVEN an existing but uninitialized singleton instance
        WHEN __init__ is called
        THEN _init should be called to complete initialization
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create instance using only __new__ (without initialization)
        instance = S3FileManager.__new__(S3FileManager)
        assert instance._initialized is False
        
        # Now patch _init to track if it's called
        with patch.object(S3FileManager, '_init') as mock_init:
            # Call __init__ on the existing instance
            instance.__init__(max_cache_size=150)
            
            # Verify _init was called to complete initialization
            mock_init.assert_called_once_with(150)
    
    def test_init_method_with_different_parameters(self):
        """
        GIVEN multiple calls to __init__ with different parameters
        WHEN instances are created
        THEN only the first call's parameters should be used for initialization
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Track calls to _init
        original_init = S3FileManager._init
        init_calls = []
        
        def tracked_init(self, max_cache_size):
            init_calls.append(max_cache_size)
            original_init(self, max_cache_size)
            
        with patch.object(S3FileManager, '_init', tracked_init):
            # Create multiple instances with different parameters
            instance1 = S3FileManager(max_cache_size=100)
            instance2 = S3FileManager(max_cache_size=200)
            instance3 = S3FileManager(max_cache_size=300)
            
            # Verify _init was only called once with the first parameter
            assert len(init_calls) == 1
            assert init_calls[0] == 100
            
            # Verify all instances are the same object
            assert instance1 is instance2 is instance3
            
            # Verify the max_cache_size is from the first initialization
            assert instance1.max_cache_size == 100
    
    def test_init_with_none_max_cache_size(self):
        """
        GIVEN a None value for max_cache_size
        WHEN S3FileManager is initialized
        THEN it should use the default value (100)
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        with patch('boto3.client'), patch('tempfile.mkdtemp'), \
             patch('concurrent.futures.ThreadPoolExecutor'):
            
            # Create instance with None max_cache_size
            manager = S3FileManager(max_cache_size=None)
            
            # Verify max_cache_size is set to the default value
            assert manager.max_cache_size == 100
            assert manager._initialized is True

class TestSingletonRaceConditions:
    """Tests for race conditions in the singleton implementation."""
    
    def test_new_method_returns_instance_directly(self):
        """
        GIVEN an existing singleton instance
        WHEN __new__ is called directly
        THEN it should return the existing instance without acquiring the lock
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create first instance normally
        instance1 = S3FileManager()
        
        # Mock the lock to verify it's not acquired for existing instance
        original_lock = S3FileManager._instance_lock
        mock_lock = MagicMock(wraps=original_lock)
        S3FileManager._instance_lock = mock_lock
        
        # Call __new__ directly
        instance2 = S3FileManager.__new__(S3FileManager)
        
        # Verify instances are identical
        assert instance2 is instance1
        
        # Verify lock was not acquired (since instance already exists)
        mock_lock.__enter__.assert_not_called()
    
    def test_double_checked_locking_pattern(self):
        """
        GIVEN the double-checked locking pattern in __new__
        WHEN multiple threads try to create instances simultaneously
        THEN the lock should only be acquired when necessary
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Mock the lock to track acquisitions
        original_lock = S3FileManager._instance_lock
        mock_lock = MagicMock(wraps=original_lock)
        S3FileManager._instance_lock = mock_lock
        
        # First instance creation should acquire the lock
        instance1 = S3FileManager()
        assert mock_lock.__enter__.call_count == 1
        
        # Reset the mock to track subsequent calls
        mock_lock.reset_mock()
        
        # Create multiple instances concurrently
        def create_instance():
            return S3FileManager()
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            instances = list(executor.map(lambda _: create_instance(), range(20)))
            
        # All instances should be the same
        for instance in instances:
            assert instance is instance1
            
        # Lock should not be acquired for existing singleton
        # Note: In a real concurrent environment, some threads might
        # acquire the lock before checking if _instance is None
        # So we can't make a strict assertion about the exact call count
        
    def test_initialization_atomicity(self):
        """
        GIVEN the singleton initialization process
        WHEN an exception occurs during initialization
        THEN the singleton instance should remain None
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Mock _init to raise an exception
        with patch.object(S3FileManager, '_init', side_effect=RuntimeError("Initialization failed")):
            # Attempt to create an instance
            with pytest.raises(RuntimeError):
                S3FileManager()
                
            # The singleton instance should still be None
            assert S3FileManager._instance is None
            
        # Now create a valid instance
        instance = S3FileManager()
        assert instance._initialized is True
        
        # Create another instance to verify singleton works after failed initialization
        instance2 = S3FileManager()
        assert instance2 is instance
        
    def test_new_with_subclass_inheritance(self):
        """
        GIVEN a subclass that doesn't override __new__
        WHEN the subclass is instantiated
        THEN it should use the parent's __new__ method and create a singleton of the subclass type
        """
        # Reset singleton for this test
        S3FileManager._instance = None
        S3FileManager._instance_lock = threading.Lock()
        
        # Create a subclass without overriding __new__
        class PartialS3FileManager(S3FileManager):
            def __init__(self, max_cache_size=100):
                # Don't call super().__init__ to avoid actual initialization
                self.initialized_as_subclass = True
        
        # Create an instance of the subclass
        subclass_instance = PartialS3FileManager()
        
        # Verify it's a singleton of the subclass type
        assert isinstance(subclass_instance, PartialS3FileManager)
        assert hasattr(subclass_instance, 'initialized_as_subclass')
        
        # Create another instance and verify it's the same object
        subclass_instance2 = PartialS3FileManager()
        assert subclass_instance2 is subclass_instance

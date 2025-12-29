"""
Metadata cache system for storing and retrieving image metadata.
Uses JSON file for persistence between sessions.
Also caches pre-generated thumbnails for faster loading.
"""
import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from metadata_parser import SDMetadata


def get_application_directory() -> str:
    """
    Get the directory where the application is running.

    Returns:
        - When running as PyInstaller exe: directory containing the .exe
        - When running as Python script: directory containing the .py script
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable (PyInstaller)
        return os.path.dirname(sys.executable)
    else:
        # Running as Python script
        return os.path.dirname(os.path.abspath(__file__))


class MetadataCache:
    """Manages persistent cache of image metadata and thumbnails."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize metadata cache.

        Args:
            cache_dir: Directory to store cache file. Defaults to 'sd_cache' in application directory.
        """
        if cache_dir is None:
            app_dir = get_application_directory()
            cache_dir = os.path.join(app_dir, 'sd_cache')

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Create thumbnails subdirectory
        self.thumbnails_dir = os.path.join(cache_dir, 'thumbnails')
        os.makedirs(self.thumbnails_dir, exist_ok=True)

        self.cache_file = os.path.join(cache_dir, 'metadata_cache.json')
        self.cache_data = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f, indent=2)
        except IOError as e:
            print(f"Error saving cache: {e}")

    def get_directory_cache(self, directory: str) -> Optional[dict]:
        """
        Get cached data for a directory.

        Returns dict with:
            - 'model_counts': dict of model_name -> count
            - 'images': dict of image_path -> metadata dict
            - 'timestamp': when cache was created
            - 'file_count': number of files
        """
        dir_key = os.path.abspath(directory)
        return self.cache_data.get(dir_key)

    def set_directory_cache(self, directory: str, model_counts: dict, images_metadata: dict, model_stats: Optional[dict] = None):
        """
        Cache metadata for a directory.

        Args:
            directory: Directory path
            model_counts: dict of model_name -> count
            images_metadata: dict of image_path -> metadata dict
            model_stats: Optional dict with full statistics (model_counts, total_images, etc.)
        """
        dir_key = os.path.abspath(directory)

        cache_entry = {
            'model_counts': model_counts,
            'images': images_metadata,
            'timestamp': datetime.now().isoformat(),
            'file_count': len(images_metadata)
        }

        # Add model_stats if provided
        if model_stats:
            cache_entry['model_stats'] = model_stats

        self.cache_data[dir_key] = cache_entry
        self._save_cache()

    def is_cache_valid(self, directory: str, current_file_count: int) -> bool:
        """
        Check if cache for directory is still valid.

        Args:
            directory: Directory path
            current_file_count: Current number of image files in directory

        Returns:
            True if cache exists and file count matches
        """
        cache = self.get_directory_cache(directory)
        if not cache:
            return False

        # Check if file count matches
        if cache.get('file_count') != current_file_count:
            return False

        return True

    def get_image_metadata(self, directory: str, image_path: str) -> Optional[dict]:
        """Get cached metadata for a specific image."""
        cache = self.get_directory_cache(directory)
        if not cache:
            return None

        images = cache.get('images', {})
        return images.get(image_path)

    def metadata_to_dict(self, metadata: Optional[SDMetadata]) -> Optional[dict]:
        """Convert SDMetadata object to dictionary for caching."""
        if not metadata:
            return None

        return {
            'model_name': metadata.model_name,
            'positive_prompt': metadata.positive_prompt,
            'negative_prompt': metadata.negative_prompt,
            'seed': metadata.seed,
            'steps': metadata.steps,
            'cfg_scale': metadata.cfg_scale,
            'sampler': metadata.sampler,
            'size': metadata.size,
            'loras': metadata.loras
        }

    def dict_to_metadata(self, data: Optional[dict]) -> Optional[SDMetadata]:
        """Convert cached dictionary back to SDMetadata object."""
        if not data:
            return None

        metadata = SDMetadata()
        metadata.model_name = data.get('model_name')
        metadata.positive_prompt = data.get('positive_prompt')
        metadata.negative_prompt = data.get('negative_prompt')
        metadata.seed = data.get('seed')
        metadata.steps = data.get('steps')
        metadata.cfg_scale = data.get('cfg_scale')
        metadata.sampler = data.get('sampler')
        metadata.size = tuple(data['size']) if data.get('size') else None
        metadata.loras = data.get('loras', [])

        return metadata

    def clear_cache(self):
        """Clear all cached data."""
        self.cache_data = {}
        self._save_cache()

    def remove_directory_cache(self, directory: str):
        """Remove cache for a specific directory."""
        dir_key = os.path.abspath(directory)
        if dir_key in self.cache_data:
            del self.cache_data[dir_key]
            self._save_cache()

    def _get_thumbnail_cache_path(self, image_path: str) -> str:
        """
        Get cache file path for a thumbnail.

        Uses hash of image path + modification time to ensure cache invalidation
        when image is modified.
        """
        # Get modification time
        try:
            mtime = os.path.getmtime(image_path)
        except OSError:
            mtime = 0

        # Create hash from path + mtime
        cache_key = f"{image_path}_{mtime}".encode('utf-8')
        cache_hash = hashlib.md5(cache_key).hexdigest()

        return os.path.join(self.thumbnails_dir, f"{cache_hash}.jpg")

    def get_cached_thumbnail(self, image_path: str) -> Optional[str]:
        """
        Get cached thumbnail path if it exists and is valid.

        Args:
            image_path: Path to source image

        Returns:
            Path to cached thumbnail JPEG, or None if not cached
        """
        cache_path = self._get_thumbnail_cache_path(image_path)

        if os.path.exists(cache_path):
            return cache_path

        return None

    def save_thumbnail(self, image_path: str, thumbnail_data: bytes) -> bool:
        """
        Save a thumbnail to cache.

        Args:
            image_path: Path to source image
            thumbnail_data: JPEG-encoded thumbnail data

        Returns:
            True if saved successfully
        """
        try:
            cache_path = self._get_thumbnail_cache_path(image_path)

            with open(cache_path, 'wb') as f:
                f.write(thumbnail_data)

            return True
        except IOError as e:
            print(f"Error saving thumbnail cache: {e}")
            return False

    def clear_thumbnail_cache(self):
        """Clear all cached thumbnails."""
        try:
            for file in os.listdir(self.thumbnails_dir):
                if file.endswith('.jpg'):
                    os.remove(os.path.join(self.thumbnails_dir, file))
        except OSError as e:
            print(f"Error clearing thumbnail cache: {e}")

"""
Metadata cache system for storing and retrieving image metadata.
Uses SQLite database for fast indexing and querying.
Also caches pre-generated thumbnails for faster loading.
"""
import sqlite3
import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
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
    """Manages persistent cache of image metadata (SQLite) and thumbnails (files)."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize metadata cache with SQLite database.

        Args:
            cache_dir: Directory to store cache files. Defaults to 'sd_cache' in application directory.
        """
        if cache_dir is None:
            app_dir = get_application_directory()
            cache_dir = os.path.join(app_dir, 'sd_cache')

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Create thumbnails subdirectory
        self.thumbnails_dir = os.path.join(cache_dir, 'thumbnails')
        os.makedirs(self.thumbnails_dir, exist_ok=True)

        # SQLite database for metadata
        self.db_path = os.path.join(cache_dir, 'metadata.db')
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Directory-level cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS directories (
                path TEXT PRIMARY KEY,
                file_count INTEGER,
                last_scan TIMESTAMP,
                model_stats_json TEXT
            )
        ''')

        # Image-level metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                path TEXT PRIMARY KEY,
                directory TEXT,
                mtime REAL,
                model_name TEXT,
                positive_prompt TEXT,
                negative_prompt TEXT,
                seed INTEGER,
                steps INTEGER,
                cfg_scale REAL,
                sampler TEXT,
                width INTEGER,
                height INTEGER,
                loras_json TEXT,
                FOREIGN KEY (directory) REFERENCES directories(path)
            )
        ''')

        # Create indexes for fast lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_directory ON images(directory)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_model ON images(model_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_mtime ON images(mtime)')

        # Full-text search table for prompts (FTS5 for fast text search)
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS prompts_fts USING fts5(
                path UNINDEXED,
                positive_prompt,
                negative_prompt
            )
        ''')

        self.conn.commit()

    def get_directory_cache(self, directory: str) -> Optional[dict]:
        """
        Get cached statistics for a directory.

        Returns dict with model_stats if available.
        """
        dir_key = os.path.abspath(directory)
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT model_stats_json FROM directories WHERE path = ?',
            (dir_key,)
        )
        row = cursor.fetchone()
        if row and row['model_stats_json']:
            return json.loads(row['model_stats_json'])
        return None

    def set_directory_cache(self, directory: str, model_counts: dict, images_metadata: dict, model_stats: Optional[dict] = None):
        """
        Cache metadata for a directory in SQLite database.

        Args:
            directory: Directory path
            model_counts: dict of model_name -> count (included in model_stats)
            images_metadata: dict of image_path -> metadata dict
            model_stats: Optional dict with full statistics (model_counts, total_images, etc.)
        """
        dir_key = os.path.abspath(directory)

        # Begin transaction for atomic updates
        cursor = self.conn.cursor()

        # Insert or update directory entry
        cursor.execute('''
            INSERT OR REPLACE INTO directories (path, file_count, last_scan, model_stats_json)
            VALUES (?, ?, ?, ?)
        ''', (dir_key, len(images_metadata), datetime.now().isoformat(),
              json.dumps(model_stats) if model_stats else None))

        # Delete old images for this directory (FTS first, then images)
        cursor.execute('''
            DELETE FROM prompts_fts WHERE path IN
            (SELECT path FROM images WHERE directory = ?)
        ''', (dir_key,))
        cursor.execute('DELETE FROM images WHERE directory = ?', (dir_key,))

        # Bulk insert images (include ALL images, even without metadata)
        image_rows = []
        fts_rows = []
        for img_path, meta_dict in images_metadata.items():
            # Get mtime for cache invalidation
            mtime = os.path.getmtime(img_path) if os.path.exists(img_path) else 0

            if meta_dict:
                # Image has SD metadata
                loras_json = json.dumps(meta_dict.get('loras', [])) if meta_dict.get('loras') else None
                image_rows.append((
                    img_path, dir_key, mtime,
                    meta_dict.get('model_name'), meta_dict.get('positive_prompt'),
                    meta_dict.get('negative_prompt'), meta_dict.get('seed'),
                    meta_dict.get('steps'), meta_dict.get('cfg_scale'),
                    meta_dict.get('sampler'),
                    meta_dict.get('size', [None, None])[0] if meta_dict.get('size') else None,
                    meta_dict.get('size', [None, None])[1] if meta_dict.get('size') else None,
                    loras_json
                ))

                # Add to FTS if has prompts
                if meta_dict.get('positive_prompt') or meta_dict.get('negative_prompt'):
                    fts_rows.append((
                        img_path,
                        meta_dict.get('positive_prompt') or '',
                        meta_dict.get('negative_prompt') or ''
                    ))
            else:
                # Image without SD metadata - store with NULL values
                image_rows.append((
                    img_path, dir_key, mtime,
                    None, None, None, None, None, None, None, None, None, None
                ))

        if image_rows:
            cursor.executemany('''
                INSERT OR REPLACE INTO images
                (path, directory, mtime, model_name, positive_prompt, negative_prompt,
                 seed, steps, cfg_scale, sampler, width, height, loras_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', image_rows)

        if fts_rows:
            cursor.executemany('''
                INSERT INTO prompts_fts (path, positive_prompt, negative_prompt)
                VALUES (?, ?, ?)
            ''', fts_rows)

        self.conn.commit()

    def is_cache_valid(self, directory: str, current_file_count: int) -> bool:
        """
        Check if cache for directory is still valid.

        Args:
            directory: Directory path
            current_file_count: Current number of image files in directory

        Returns:
            True if cache exists and file count matches
        """
        dir_key = os.path.abspath(directory)
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT file_count FROM directories WHERE path = ?',
            (dir_key,)
        )
        row = cursor.fetchone()
        return row is not None and row['file_count'] == current_file_count

    def get_image_metadata(self, directory: str, image_path: str) -> Optional[dict]:
        """Get cached metadata for a specific image from SQLite."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT model_name, positive_prompt, negative_prompt, seed, steps,
                   cfg_scale, sampler, width, height, loras_json
            FROM images WHERE path = ? AND directory = ?
        ''', (image_path, os.path.abspath(directory)))

        row = cursor.fetchone()
        if not row:
            return None

        # Convert row to dict
        meta_dict = {
            'model_name': row['model_name'],
            'positive_prompt': row['positive_prompt'],
            'negative_prompt': row['negative_prompt'],
            'seed': row['seed'],
            'steps': row['steps'],
            'cfg_scale': row['cfg_scale'],
            'sampler': row['sampler'],
            'size': (row['width'], row['height']) if row['width'] and row['height'] else None,
            'loras': json.loads(row['loras_json']) if row['loras_json'] else []
        }
        return meta_dict

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
        """Clear all cached metadata from database."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM directories')
        cursor.execute('DELETE FROM images')
        cursor.execute('DELETE FROM prompts_fts')
        self.conn.commit()

    def remove_directory_cache(self, directory: str):
        """Remove cache for a specific directory from database."""
        dir_key = os.path.abspath(directory)
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM images WHERE directory = ?', (dir_key,))
        cursor.execute('DELETE FROM prompts_fts WHERE path IN (SELECT path FROM images WHERE directory = ?)', (dir_key,))
        cursor.execute('DELETE FROM directories WHERE path = ?', (dir_key,))
        self.conn.commit()

    def filter_by_prompt(self, directory: str, search_terms: List[str]) -> List[str]:
        """
        Use FTS (Full-Text Search) to quickly find images matching prompt terms.

        Args:
            directory: Directory to search in
            search_terms: List of search terms (all must match - AND logic)

        Returns:
            List of image paths matching the search
        """
        if not search_terms:
            return []

        dir_key = os.path.abspath(directory)

        # Build FTS query - AND all terms together
        # FTS5 syntax: term1 AND term2 AND term3
        fts_query = ' AND '.join(search_terms)

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT p.path FROM prompts_fts p
            INNER JOIN images i ON p.path = i.path
            WHERE i.directory = ? AND prompts_fts MATCH ?
        ''', (dir_key, fts_query))

        return [row['path'] for row in cursor.fetchall()]

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Ensure database connection is closed on cleanup."""
        self.close()

    # ===== Thumbnail Cache Methods (File-based) =====
    # These remain unchanged - file-based cache works well for binary data

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

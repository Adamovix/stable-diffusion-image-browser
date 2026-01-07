"""
Filter engine with pre-computed indices for instant filtering.
Uses set operations and SQLite FTS for fast lookups.
"""
from typing import Optional, List, Tuple, Set, Dict
from collections import defaultdict


class FilterEngine:
    """
    Pre-computes filter indices for instant filtering operations.
    Builds indices once per directory, then filters using set operations.
    """

    def __init__(self, directory: str, metadata_cache: Optional[object] = None):
        """
        Initialize filter engine for a directory.

        Args:
            directory: Directory path to build indices for
            metadata_cache: MetadataCache instance for database access
        """
        self.directory = directory
        self.cache = metadata_cache

        # Pre-computed filter indices (built once, reused many times)
        self.all_images: List[str] = []  # All image paths
        self.images_with_metadata: Set[str] = set()  # Has SD metadata (prompts)
        self.images_by_model: Dict[str, Set[str]] = defaultdict(set)  # model_name -> {paths}
        self.images_without_model: Set[str] = set()  # Has prompt but no model
        self.metadata_by_path: Dict[str, dict] = {}  # path -> metadata dict

        # Build indices from cache
        self._build_indices()

    def _build_indices(self):
        """Build filter indices from cached metadata (fast - single SQL query)."""
        if not self.cache or not hasattr(self.cache, 'conn'):
            return

        import os
        dir_key = os.path.abspath(self.directory)

        # Single SQL query to get all images in directory
        cursor = self.cache.conn.cursor()
        cursor.execute('''
            SELECT path, model_name, positive_prompt, negative_prompt,
                   seed, steps, cfg_scale, sampler, width, height, loras_json
            FROM images WHERE directory = ?
            ORDER BY mtime DESC
        ''', (dir_key,))

        row_count = 0

        for row in cursor.fetchall():
            row_count += 1
            path = row['path']
            model = row['model_name']
            prompt = row['positive_prompt']

            # Store in all images list (includes images without metadata)
            self.all_images.append(path)

            # Build metadata dict for this image (even if mostly None)
            self.metadata_by_path[path] = {
                'model_name': model,
                'positive_prompt': prompt,
                'negative_prompt': row['negative_prompt'],
                'seed': row['seed'],
                'steps': row['steps'],
                'cfg_scale': row['cfg_scale'],
                'sampler': row['sampler'],
                'size': (row['width'], row['height']) if row['width'] and row['height'] else None,
                'loras': []  # Will be populated if needed
            }

            # Build filter indices (only for images WITH metadata)
            if prompt:
                self.images_with_metadata.add(path)

                if model:
                    self.images_by_model[model].add(path)
                else:
                    self.images_without_model.add(path)

    def apply_filters(self, model_filter: str, prompt_filter: str) -> List[Tuple[str, dict]]:
        """
        Apply filters using pre-computed indices (instant - O(1) lookups).

        Args:
            model_filter: Model filter string or special filter code
            prompt_filter: Comma-separated prompt search terms

        Returns:
            List of (path, metadata) tuples matching filters
        """
        # Step 1: Model filter (set operations - O(1))
        if model_filter == "__ALL_IMAGES__":
            candidates = set(self.all_images)
        elif model_filter == "__ALL_MODELS__":
            candidates = self.images_with_metadata.copy()
        elif model_filter == "__UNKNOWN_MODEL__":
            candidates = self.images_without_model.copy()
        elif model_filter:
            # Specific model filter
            candidates = self.images_by_model.get(model_filter, set()).copy()
        else:
            candidates = set(self.all_images)

        # Step 2: Prompt filter (use FTS if available, else fallback)
        if prompt_filter and self.cache:
            search_terms = [term.strip().lower() for term in prompt_filter.split(',') if term.strip()]

            if search_terms:
                # Use SQLite FTS for fast full-text search
                try:
                    matching_paths = set(self.cache.filter_by_prompt(self.directory, search_terms))
                    candidates &= matching_paths
                except Exception as e:
                    # Fallback to Python filtering if FTS fails
                    print(f"FTS search failed, using fallback: {e}")
                    matching_paths = self._filter_prompts_fallback(candidates, search_terms)
                    candidates &= matching_paths

        # Step 3: Build result list with metadata
        result = []
        for path in candidates:
            metadata = self.metadata_by_path.get(path)
            if metadata:
                result.append((path, metadata))

        return result

    def _filter_prompts_fallback(self, candidates: Set[str], search_terms: List[str]) -> Set[str]:
        """
        Fallback prompt filtering using Python (if FTS unavailable).

        Args:
            candidates: Set of image paths to filter
            search_terms: List of search terms (all must match)

        Returns:
            Set of paths matching all search terms
        """
        matching = set()

        for path in candidates:
            metadata = self.metadata_by_path.get(path)
            if not metadata:
                continue

            prompt_text = str(metadata.get('positive_prompt', '')).lower()

            # Check if all terms are present (AND logic)
            if all(term in prompt_text for term in search_terms):
                matching.add(path)

        return matching

    def get_stats(self) -> dict:
        """Get statistics about indexed images."""
        return {
            'total_images': len(self.all_images),
            'images_with_metadata': len(self.images_with_metadata),
            'images_without_model': len(self.images_without_model),
            'unique_models': len(self.images_by_model),
            'model_counts': {
                model: len(paths) for model, paths in self.images_by_model.items()
            }
        }

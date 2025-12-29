"""
Stable Diffusion Image Browser
Main application window with directory tree, image viewer, and metadata display.
"""
import sys
import os
from pathlib import Path
from typing import List, Optional
from collections import Counter

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeView, QScrollArea, QLabel, QTextEdit, QPushButton,
    QLineEdit, QGridLayout, QFrame, QComboBox,
    QGroupBox, QSizePolicy, QProgressDialog, QTabWidget, QProgressBar
)
from PyQt6.QtCore import Qt, QDir, QSize, QThread, pyqtSignal, QModelIndex, QTimer, QUrl, QMimeData
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFileSystemModel, QDrag, QDesktopServices

from PIL import Image

from metadata_parser import extract_metadata, get_metadata_summary, SDMetadata
from metadata_cache import MetadataCache


class ImageThumbnail(QFrame):
    """Widget representing a single image thumbnail in the grid."""

    clicked = pyqtSignal(str)  # Emits file path when clicked

    def __init__(self, image_path: str, metadata_cache: Optional['MetadataCache'] = None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.metadata: Optional[SDMetadata] = None
        self.metadata_cache = metadata_cache
        self.is_selected = False
        self.thumbnail_loaded = False
        self.drag_start_position = None

        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Thumbnail image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setMaximumSize(200, 200)
        self.image_label.setText("Loading...")  # Placeholder text

        # Filename label
        self.name_label = QLabel(Path(image_path).name)
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setMaximumWidth(200)

        layout.addWidget(self.image_label)
        layout.addWidget(self.name_label)
        self.setLayout(layout)

        # Don't load thumbnail immediately - will be loaded later by ImageGridView

    def load_thumbnail(self):
        """Load and display thumbnail of the image."""
        if self.thumbnail_loaded:
            return  # Already loaded

        try:
            # Try to load from cache first
            if self.metadata_cache:
                cached_thumb = self.metadata_cache.get_cached_thumbnail(self.image_path)
                if cached_thumb:
                    # Load cached JPEG thumbnail (MUCH faster!)
                    pixmap = QPixmap(cached_thumb)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(
                            200, 200,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        self.image_label.setPixmap(scaled_pixmap)
                        self.thumbnail_loaded = True
                        return

            # Cache miss or no cache - generate thumbnail
            # Use PIL to load image and create thumbnail
            with Image.open(self.image_path) as img:
                # Convert to RGB if needed
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                # Calculate thumbnail size while maintaining aspect ratio
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)

                # Save to cache as JPEG
                if self.metadata_cache:
                    import io
                    buffer = io.BytesIO()
                    # Convert RGBA to RGB for JPEG
                    if img.mode == "RGBA":
                        # Create white background
                        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3])  # Use alpha as mask
                        rgb_img.save(buffer, format='JPEG', quality=85, optimize=True)
                    else:
                        img.save(buffer, format='JPEG', quality=85, optimize=True)

                    self.metadata_cache.save_thumbnail(self.image_path, buffer.getvalue())

                # Convert PIL image to QPixmap with proper stride
                if img.mode == "RGB":
                    data = img.tobytes("raw", "RGB")
                    bytes_per_line = img.width * 3
                    qimage = QImage(data, img.width, img.height, bytes_per_line, QImage.Format.Format_RGB888)
                elif img.mode == "RGBA":
                    data = img.tobytes("raw", "RGBA")
                    bytes_per_line = img.width * 4
                    qimage = QImage(data, img.width, img.height, bytes_per_line, QImage.Format.Format_RGBA8888)

                pixmap = QPixmap.fromImage(qimage)

                # Scale pixmap to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    200, 200,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.thumbnail_loaded = True

        except Exception as e:
            self.image_label.setText(f"Error loading\n{Path(self.image_path).name}")
            print(f"Error loading thumbnail {self.image_path}: {e}")
            self.thumbnail_loaded = True  # Mark as loaded even if failed to avoid retrying

    def set_selected(self, selected: bool):
        """Set the selection state of this thumbnail."""
        self.is_selected = selected
        if selected:
            self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
            self.setLineWidth(3)
            self.setStyleSheet("background-color: #e3f2fd; border: 3px solid #2196f3;")
        else:
            self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
            self.setLineWidth(2)
            self.setStyleSheet("")

    def mousePressEvent(self, event):
        """Handle mouse click to select image."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = event.pos()
            self.clicked.emit(self.image_path)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to open image in default application."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Open the image file with the default system application
            file_url = QUrl.fromLocalFile(self.image_path)
            QDesktopServices.openUrl(file_url)
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse drag to initiate drag and drop operation."""
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self.drag_start_position is None:
            return

        # Check if drag distance is sufficient (prevents accidental drags on clicks)
        if (event.pos() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return

        # Create drag operation
        drag = QDrag(self)
        mime_data = QMimeData()

        # Set file URL for drag and drop to other applications
        file_url = QUrl.fromLocalFile(self.image_path)
        mime_data.setUrls([file_url])

        drag.setMimeData(mime_data)

        # Optional: Set drag pixmap (small preview of the image being dragged)
        if self.thumbnail_loaded and self.image_label.pixmap():
            pixmap = self.image_label.pixmap()
            # Scale down for drag cursor if it's too large
            if pixmap.width() > 128 or pixmap.height() > 128:
                pixmap = pixmap.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            drag.setPixmap(pixmap)

        # Execute the drag
        drag.exec(Qt.DropAction.CopyAction)

        super().mouseMoveEvent(event)


class ImageGridView(QScrollArea):
    """Scrollable grid view for displaying image thumbnails."""

    image_selected = pyqtSignal(str)  # Emits selected image path
    progress_show = pyqtSignal(str, int)  # Emits (message, maximum)
    progress_update = pyqtSignal(int, str)  # Emits (value, message)
    progress_hide = pyqtSignal()  # Emits when progress should hide

    def __init__(self, metadata_cache: Optional['MetadataCache'] = None, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Container widget for grid
        self.container = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(10)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.container.setLayout(self.grid_layout)
        self.setWidget(self.container)

        self.thumbnails: List[ImageThumbnail] = []
        self.current_directory: Optional[str] = None
        self.selected_thumbnail: Optional[ImageThumbnail] = None
        self.metadata_cache = metadata_cache  # Cache for faster filtering

        # Progressive loading
        self.loading_index = 0
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self._load_next_batch)

        # Dynamic column calculation
        self.column_count = 4  # Default, will be recalculated
        self.thumbnail_width = 220  # 200px thumbnail + 20px margin
        self.min_columns = 1  # Minimum number of columns
        self.max_columns = 8  # Maximum number of columns

    def calculate_columns(self) -> int:
        """Calculate optimal number of columns based on available width."""
        # Get available width (viewport width minus scrollbar if present)
        available_width = self.viewport().width()

        # Calculate how many columns can fit
        columns = max(self.min_columns, available_width // self.thumbnail_width)

        # Cap at maximum
        columns = min(columns, self.max_columns)

        return columns

    def resizeEvent(self, event):
        """Handle resize events to recalculate columns."""
        super().resizeEvent(event)

        # Recalculate column count
        new_column_count = self.calculate_columns()

        # If column count changed and we have thumbnails, reorganize grid
        if new_column_count != self.column_count and self.thumbnails:
            self.column_count = new_column_count
            self._reorganize_grid()

    def _reorganize_grid(self):
        """Reorganize thumbnails in grid with new column count."""
        # Remove all widgets from grid (but don't delete them)
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)

        # Re-add thumbnails with new column layout
        for idx, thumbnail in enumerate(self.thumbnails):
            row = idx // self.column_count
            col = idx % self.column_count
            self.grid_layout.addWidget(thumbnail, row, col)

        self.container.adjustSize()

    def clear_grid(self):
        """Clear all thumbnails from the grid."""
        # Stop any ongoing loading
        self.loading_timer.stop()
        self.loading_index = 0

        for thumbnail in self.thumbnails:
            thumbnail.deleteLater()
        self.thumbnails.clear()
        self.selected_thumbnail = None  # Clear selection reference

        # Process deletion events immediately to clear visual display
        QApplication.processEvents()

    def _load_next_batch(self):
        """Load the next batch of thumbnails progressively (legacy method for compatibility)."""
        # Batch size - load this many thumbnails at a time
        batch_size = 5

        # Calculate end index for this batch
        end_index = min(self.loading_index + batch_size, len(self.thumbnails))

        # Load thumbnails in this batch
        for i in range(self.loading_index, end_index):
            self.thumbnails[i].load_thumbnail()

        # Update loading index
        self.loading_index = end_index

        # Stop timer if all thumbnails are loaded
        if self.loading_index >= len(self.thumbnails):
            self.loading_timer.stop()

    def _load_thumbnails_progressive(self, filtered_files: list):
        """
        Fast progressive loading for cached thumbnails.
        Creates widgets and loads thumbnails in small batches without blocking.
        Gives responsive, streaming feel.
        """
        # Store filtered files for progressive loading
        self.pending_files = filtered_files
        self.loading_index = 0

        # Show progress bar
        self.progress_show.emit("Loading thumbnails", len(filtered_files))

        # Start timer for progressive loading
        self.loading_timer.stop()
        try:
            self.loading_timer.timeout.disconnect()
        except:
            pass  # No connections to disconnect
        self.loading_timer.timeout.connect(self._load_next_progressive_batch)
        self.loading_timer.start(10)  # Very fast, 10ms between batches

    def _load_next_progressive_batch(self):
        """Load next batch of thumbnails progressively (widget creation + thumbnail loading)."""
        if not hasattr(self, 'pending_files') or self.loading_index >= len(self.pending_files):
            self.loading_timer.stop()
            self.container.adjustSize()
            self.progress_hide.emit()  # Hide progress when done
            return

        # Batch size: create and load this many at once
        # Smaller = more responsive, larger = faster overall
        batch_size = 10

        end_idx = min(self.loading_index + batch_size, len(self.pending_files))

        for i in range(self.loading_index, end_idx):
            img_path, metadata = self.pending_files[i]

            row = i // self.column_count
            col = i % self.column_count

            # Create widget
            thumbnail = ImageThumbnail(img_path, self.metadata_cache)
            thumbnail.metadata = metadata
            thumbnail.clicked.connect(self._on_thumbnail_clicked)

            # Add to grid and list
            self.grid_layout.addWidget(thumbnail, row, col)
            self.thumbnails.append(thumbnail)

            # Load thumbnail immediately (fast since cached)
            thumbnail.load_thumbnail()

        self.loading_index = end_idx

        # Update progress
        self.progress_update.emit(end_idx, "Loading thumbnails")

        # Process events to keep UI responsive
        QApplication.processEvents()

    def _load_thumbnails_with_progress(self, filtered_files: list):
        """
        Load thumbnails with progress bar for first-time load.
        Creates widgets and loads thumbnails together in batches for immediate visual feedback.
        """
        # Show progress for loading thumbnails
        self.progress_show.emit("Loading thumbnails", len(filtered_files))

        # Create and load thumbnails in batches for immediate feedback
        batch_size = 5  # Create and load this many at once

        for idx, (img_path, metadata) in enumerate(filtered_files):
            # Update progress at start of each batch
            if idx % batch_size == 0:
                self.progress_update.emit(idx, "Loading thumbnails")

            row = idx // self.column_count
            col = idx % self.column_count

            # Create widget
            thumbnail = ImageThumbnail(img_path, self.metadata_cache)
            thumbnail.metadata = metadata
            thumbnail.clicked.connect(self._on_thumbnail_clicked)

            self.grid_layout.addWidget(thumbnail, row, col)
            self.thumbnails.append(thumbnail)

            # Load thumbnail immediately (create + load together)
            thumbnail.load_thumbnail()

            # Process events at end of each batch to show progress
            if (idx + 1) % batch_size == 0:
                QApplication.processEvents()

        # Update progress to 100%
        self.progress_update.emit(len(filtered_files), "Loading thumbnails")

        # Update layout
        self.container.adjustSize()

        # Hide progress when done
        self.progress_hide.emit()

    def load_images(self, directory: str, filter_model: str = "", filter_prompt: str = ""):
        """Load images from directory and display in grid."""
        try:
            self.clear_grid()
            # Process any pending delete events before proceeding
            QApplication.processEvents()
            self.current_directory = directory

            if not os.path.isdir(directory):
                return

            # Supported image formats
            extensions = ['.png', '.jpg', '.jpeg', '.webp']
            image_files_set = set()  # Use set to avoid duplicates on Windows

            for ext in extensions:
                image_files_set.update(Path(directory).glob(f"*{ext}"))
                image_files_set.update(Path(directory).glob(f"*{ext.upper()}"))

            # Convert back to list
            image_files = list(image_files_set)

            if not image_files:
                return

            # Sort by modification time (newest first)
            image_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Show progress bar for filtering
            self.progress_show.emit("Loading images", len(image_files))

            # Apply filters
            filtered_files = []
            for i, img_path in enumerate(image_files):
                # Update progress
                if i % 10 == 0:  # Update every 10 images to avoid slowdown
                    self.progress_update.emit(i, "Loading images")

                try:
                    # Try to get metadata from cache first, otherwise extract
                    metadata = None
                    if self.metadata_cache:
                        cached_meta_dict = self.metadata_cache.get_image_metadata(directory, str(img_path))
                        if cached_meta_dict:
                            metadata = self.metadata_cache.dict_to_metadata(cached_meta_dict)

                    # If not in cache, extract it
                    if not metadata:
                        metadata = extract_metadata(str(img_path))

                    # Apply model filter
                    if filter_model == "__ALL_IMAGES__":
                        # Show all images, no filtering by metadata
                        pass
                    elif filter_model == "__ALL_MODELS__":
                        # Show only images with SD prompts
                        if not metadata or not metadata.positive_prompt:
                            continue
                    elif filter_model == "__UNKNOWN_MODEL__":
                        # Show images with prompt but no model name
                        if not metadata or not metadata.positive_prompt:
                            continue
                        if metadata.model_name:
                            continue
                    elif filter_model:
                        # Specific model filter
                        if not metadata or not metadata.model_name:
                            continue
                        if filter_model.lower() not in metadata.model_name.lower():
                            continue

                    # Apply prompt filter (supports comma-separated terms - all must match)
                    if filter_prompt:
                        if not metadata or not metadata.positive_prompt:
                            continue
                        # Defensive check: ensure positive_prompt is a string
                        prompt_text = str(metadata.positive_prompt) if metadata.positive_prompt else ""
                        prompt_text_lower = prompt_text.lower()

                        # Split by comma and check all terms are present
                        search_terms = [term.strip() for term in filter_prompt.split(',')]
                        search_terms = [term for term in search_terms if term]  # Remove empty strings

                        # All terms must be present (AND logic)
                        if not all(term.lower() in prompt_text_lower for term in search_terms):
                            continue

                    filtered_files.append((str(img_path), metadata))

                except Exception as e:
                    # Skip images that cause errors but don't crash
                    print(f"Error processing {img_path}: {e}")
                    continue

            # Update progress to 100%
            self.progress_update.emit(len(image_files), "Loading images")

            if not filtered_files:
                self.progress_hide.emit()
                return

            # Calculate optimal column count based on current width
            self.column_count = self.calculate_columns()

            # Check cache hit rate to decide loading strategy
            cache_hit_rate = 0.0
            if self.metadata_cache and len(filtered_files) > 0:
                cached_count = sum(1 for img_path, _ in filtered_files
                                  if self.metadata_cache.get_cached_thumbnail(img_path) is not None)
                cache_hit_rate = cached_count / len(filtered_files)

            # If most thumbnails are cached (>70%), use fast progressive loading
            # Otherwise use progress dialog for slower first-time load
            if cache_hit_rate > 0.7:
                # Fast path: Progressive loading without progress dialog
                self._load_thumbnails_progressive(filtered_files)
            else:
                # Slow path: Progress dialog for first-time load
                self._load_thumbnails_with_progress(filtered_files)

        except Exception as e:
            print(f"Error in load_images: {e}")
            import traceback
            traceback.print_exc()

    def _on_thumbnail_clicked(self, image_path: str):
        """Handle thumbnail click - update selection and emit signal."""
        # Deselect previous thumbnail (with safety check for deleted objects)
        if self.selected_thumbnail:
            try:
                self.selected_thumbnail.set_selected(False)
            except RuntimeError:
                # Thumbnail was already deleted
                pass

        # Find and select new thumbnail
        for thumbnail in self.thumbnails:
            if thumbnail.image_path == image_path:
                thumbnail.set_selected(True)
                self.selected_thumbnail = thumbnail
                break

        # Emit signal
        self.image_selected.emit(image_path)


class MetadataPanel(QWidget):
    """Panel for displaying image metadata."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Image Metadata")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Workflow tab - formatted metadata display
        self.workflow_text = QTextEdit()
        self.workflow_text.setReadOnly(True)
        self.workflow_text.setMinimumHeight(150)
        self.tab_widget.addTab(self.workflow_text, "Workflow")

        # Raw metadata tab - raw dump
        self.raw_text = QTextEdit()
        self.raw_text.setReadOnly(True)
        self.raw_text.setMinimumHeight(150)
        self.raw_text.setFont(self.workflow_text.font())  # Use monospace for raw data would be better, but keep consistent
        self.tab_widget.addTab(self.raw_text, "Raw metadata")

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def display_metadata(self, image_path: str):
        """Extract and display metadata from image."""
        from PIL import Image as PILImage

        metadata = extract_metadata(image_path)
        filename = Path(image_path).name

        # Display formatted metadata in Workflow tab
        if metadata:
            summary = get_metadata_summary(metadata)
            text = f"File: {filename}\n\n"

            for key, value in summary.items():
                text += f"{key}:\n{value}\n\n"

            self.workflow_text.setText(text)
        else:
            self.workflow_text.setText(f"File: {filename}\n\nNo metadata found.")

        # Display raw metadata in Raw metadata tab
        raw_text = f"File: {filename}\n\n"
        raw_text += "="*50 + "\n"
        raw_text += "RAW METADATA DUMP\n"
        raw_text += "="*50 + "\n\n"

        try:
            # Try to extract raw PNG chunks or EXIF data
            with PILImage.open(image_path) as img:
                if img.format == 'PNG':
                    raw_text += "PNG Text Chunks:\n"
                    raw_text += "-"*50 + "\n"
                    if hasattr(img, 'info') and img.info:
                        for key, value in img.info.items():
                            raw_text += f"\n[{key}]\n"
                            if isinstance(value, (str, bytes)):
                                value_str = value if isinstance(value, str) else value.decode('utf-8', errors='replace')
                                raw_text += f"{value_str}\n"
                            else:
                                raw_text += f"{value}\n"
                    else:
                        raw_text += "No PNG text chunks found.\n"

                elif img.format in ('JPEG', 'JPG'):
                    raw_text += "JPEG EXIF Data:\n"
                    raw_text += "-"*50 + "\n"
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = img._getexif()
                        for tag_id, value in exif.items():
                            raw_text += f"\nTag {tag_id}:\n{value}\n"
                    elif hasattr(img, 'info') and img.info:
                        for key, value in img.info.items():
                            raw_text += f"\n[{key}]\n{value}\n"
                    else:
                        raw_text += "No EXIF data found.\n"

                else:
                    raw_text += f"Format: {img.format}\n"
                    raw_text += "Image info:\n"
                    if hasattr(img, 'info') and img.info:
                        for key, value in img.info.items():
                            raw_text += f"\n[{key}]\n{value}\n"
                    else:
                        raw_text += "No metadata found.\n"

            # Also show parsed metadata object if available
            if metadata:
                raw_text += "\n\n" + "="*50 + "\n"
                raw_text += "PARSED METADATA OBJECT\n"
                raw_text += "="*50 + "\n"
                raw_text += f"\nmodel_name: {metadata.model_name}\n"
                raw_text += f"positive_prompt: {metadata.positive_prompt}\n"
                raw_text += f"negative_prompt: {metadata.negative_prompt}\n"
                raw_text += f"seed: {metadata.seed}\n"
                raw_text += f"steps: {metadata.steps}\n"
                raw_text += f"cfg_scale: {metadata.cfg_scale}\n"
                raw_text += f"sampler: {metadata.sampler}\n"
                raw_text += f"size: {metadata.size}\n"
                raw_text += f"loras: {metadata.loras}\n"
                if metadata.raw_data:
                    raw_text += f"\nraw_data (first 1000 chars):\n{metadata.raw_data[:1000]}\n"

        except Exception as e:
            raw_text += f"\nError reading raw metadata: {e}\n"

        self.raw_text.setText(raw_text)

    def clear_metadata(self):
        """Clear metadata display."""
        self.workflow_text.clear()
        self.raw_text.clear()


class FilterPanel(QWidget):
    """Panel for search and filtering controls."""

    filter_changed = pyqtSignal(str, str)  # Emits (model_filter, prompt_filter)
    refresh_requested = pyqtSignal()  # Emits when user wants to refresh cache

    def __init__(self, parent=None):
        super().__init__(parent)

        # Use horizontal layout for more compact design
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)  # Reduce vertical margins
        layout.setSpacing(8)  # Add consistent spacing between elements

        # Model filter - use dropdown instead of text input
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.setMaximumHeight(28)  # Limit height
        self.model_combo.setMinimumWidth(200)  # Ensure readable width
        self.model_combo.addItem("All Models")  # Default option
        layout.addWidget(model_label)
        layout.addWidget(self.model_combo)

        # Prompt filter
        prompt_label = QLabel("Prompt:")
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Filter by prompt (comma-separated)...")
        self.prompt_input.setMaximumHeight(28)  # Limit height
        layout.addWidget(prompt_label)
        layout.addWidget(self.prompt_input)

        # Buttons
        self.apply_button = QPushButton("Apply")
        self.apply_button.setMaximumWidth(70)
        self.apply_button.setMaximumHeight(28)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setMaximumWidth(70)
        self.clear_button.setMaximumHeight(28)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setMaximumWidth(70)
        self.refresh_button.setMaximumHeight(28)
        self.refresh_button.setToolTip("Rescan directory and rebuild cache")
        layout.addWidget(self.apply_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.refresh_button)

        self.setLayout(layout)
        self.setMaximumHeight(40)  # Limit overall panel height

        # Connect signals
        self.apply_button.clicked.connect(self.emit_filters)
        self.clear_button.clicked.connect(self.clear_filters)
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.prompt_input.returnPressed.connect(self.emit_filters)

    def on_model_changed(self, text):
        """Auto-apply filter when model selection changes."""
        self.emit_filters()

    def emit_filters(self):
        """Emit current filter values."""
        model_text = self.model_combo.currentText()

        # Handle special filter options
        if model_text.startswith("All Images"):
            # Show all images regardless of metadata
            model_text = "__ALL_IMAGES__"
        elif model_text.startswith("All Models"):
            # Show only images with SD prompts
            model_text = "__ALL_MODELS__"
        elif model_text.startswith("Unknown model"):
            # Show images with prompt but no model
            model_text = "__UNKNOWN_MODEL__"
        else:
            # Extract model name (remove count suffix like " (5)")
            if " (" in model_text:
                model_text = model_text[:model_text.rfind(" (")]

        prompt_text = self.prompt_input.text().strip()
        self.filter_changed.emit(model_text, prompt_text)

    def clear_filters(self):
        """Clear all filter inputs."""
        self.model_combo.setCurrentIndex(0)  # Select "All Images"
        self.prompt_input.clear()
        self.filter_changed.emit("__ALL_IMAGES__", "")

    def populate_models(self, model_stats: dict):
        """
        Populate the model dropdown with models and their counts.
        model_stats: dict with 'model_counts', 'total_images', 'images_with_prompt', 'images_without_model'
        """
        current_selection = self.model_combo.currentText()

        # Block signals to prevent auto-filtering during repopulation
        self.model_combo.blockSignals(True)

        # Clear and repopulate
        self.model_combo.clear()

        # Extract statistics
        model_counts = model_stats.get('model_counts', {})
        total_images = model_stats.get('total_images', 0)
        images_with_prompt = model_stats.get('images_with_prompt', 0)
        images_without_model = model_stats.get('images_without_model', 0)

        # Add "All Images" as default option
        if total_images > 0:
            self.model_combo.addItem(f"All Images ({total_images})")
        else:
            self.model_combo.addItem("All Images")

        # Add "All Models" - only images with SD prompts
        if images_with_prompt > 0:
            self.model_combo.addItem(f"All Models ({images_with_prompt})")
        else:
            self.model_combo.addItem("All Models")

        # Add "Unknown model" - images with prompt but no model
        if images_without_model > 0:
            self.model_combo.addItem(f"Unknown model ({images_without_model})")

        # Sort models alphabetically and add with counts
        sorted_models = sorted(model_counts.items())
        for model_name, count in sorted_models:
            display_text = f"{model_name} ({count})"
            self.model_combo.addItem(display_text)

        # Try to restore previous selection
        restored = False
        if current_selection and current_selection not in ["All Images", "All Models", "Unknown model"]:
            # Extract model name from previous selection
            prev_model = current_selection
            if " (" in prev_model:
                prev_model = prev_model[:prev_model.rfind(" (")]

            # Find matching item
            for i in range(self.model_combo.count()):
                item_text = self.model_combo.itemText(i)
                if " (" in item_text:
                    item_model = item_text[:item_text.rfind(" (")]
                    if item_model == prev_model:
                        self.model_combo.setCurrentIndex(i)
                        restored = True
                        break

        # If not restored, select "All Images" (default)
        if not restored:
            self.model_combo.setCurrentIndex(0)

        # Re-enable signals
        self.model_combo.blockSignals(False)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stable Diffusion Image Browser")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize metadata cache
        self.metadata_cache = MetadataCache()

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Use vertical layout to accommodate progress bar at bottom
        root_layout = QVBoxLayout()
        central_widget.setLayout(root_layout)

        # Main horizontal layout for content
        main_layout = QHBoxLayout()
        root_layout.addLayout(main_layout)

        # Create left-center container (filters + tree + images)
        left_center_container = QWidget()
        left_center_layout = QVBoxLayout()
        left_center_layout.setContentsMargins(0, 0, 0, 0)
        left_center_layout.setSpacing(2)  # Minimal spacing between filter and content

        # Add filter panel at top of left-center
        self.filter_panel = FilterPanel()
        self.filter_panel.filter_changed.connect(self.apply_filters)
        self.filter_panel.refresh_requested.connect(self.refresh_current_directory)
        left_center_layout.addWidget(self.filter_panel)

        # Create splitter for directory tree and image grid
        left_center_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Directory tree
        self.setup_directory_tree()
        left_center_splitter.addWidget(self.tree_view)

        # Center panel: Image grid view
        self.image_grid = ImageGridView(metadata_cache=self.metadata_cache)
        self.image_grid.image_selected.connect(self.on_image_selected)
        self.image_grid.progress_show.connect(self.show_progress)
        self.image_grid.progress_update.connect(self.update_progress)
        self.image_grid.progress_hide.connect(self.hide_progress)
        left_center_splitter.addWidget(self.image_grid)

        # Set initial sizes for tree and grid (20% tree, 80% images)
        left_center_splitter.setSizes([280, 700])

        left_center_layout.addWidget(left_center_splitter)
        left_center_container.setLayout(left_center_layout)

        # Add left-center container to main layout
        main_layout.addWidget(left_center_container)

        # Right panel: Selected image preview + metadata (full height)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Image preview
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumSize(400, 400)
        self.image_preview.setScaledContents(False)
        self.image_preview.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.image_preview.setText("Select an image to preview")
        right_layout.addWidget(self.image_preview)

        # Metadata display
        self.metadata_panel = MetadataPanel()
        right_layout.addWidget(self.metadata_panel)

        right_panel.setLayout(right_layout)

        # Add right panel to main layout
        main_layout.addWidget(right_panel)

        # Set stretch factors for left-center vs right (70% vs 30%)
        main_layout.setStretch(0, 7)
        main_layout.setStretch(1, 3)

        # Add progress bar at bottom
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(20)  # Narrow bar
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()  # Hidden by default
        root_layout.addWidget(self.progress_bar)

        # Store current state
        self.current_image_path: Optional[str] = None
        self.current_directory: Optional[str] = None

    def show_progress(self, message: str, maximum: int = 0):
        """Show progress bar with message and optional maximum value."""
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setFormat(f"{message} - %p%")
        self.progress_bar.show()
        QApplication.processEvents()

    def update_progress(self, value: int, message: str = None):
        """Update progress bar value and optionally change message."""
        self.progress_bar.setValue(value)
        if message:
            self.progress_bar.setFormat(f"{message} - %p%")
        QApplication.processEvents()

    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.hide()
        QApplication.processEvents()

    def setup_directory_tree(self):
        """Set up the directory tree view."""
        self.tree_view = QTreeView()
        self.file_model = QFileSystemModel()

        # Set root path to filesystem root to show all drives
        # On Windows this shows C:, D:, etc. On Linux/Mac shows /
        self.file_model.setRootPath("")

        # Only show directories
        self.file_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.Drives)

        self.tree_view.setModel(self.file_model)

        # On Windows, show "My Computer" root. On Unix, show "/"
        if sys.platform == "win32":
            # Don't set a root index - show all drives
            self.tree_view.setRootIndex(self.file_model.index(""))
        else:
            # On Unix, start at root
            self.tree_view.setRootIndex(self.file_model.index("/"))

        # Hide unnecessary columns
        self.tree_view.setColumnHidden(1, True)  # Size
        self.tree_view.setColumnHidden(2, True)  # Type
        self.tree_view.setColumnHidden(3, True)  # Date Modified

        # Connect selection signal
        self.tree_view.clicked.connect(self.on_directory_selected)

    def on_directory_selected(self, index: QModelIndex):
        """Handle directory selection in tree view."""
        directory = self.file_model.filePath(index)
        self.load_directory(directory)

    def load_directory(self, directory: str):
        """Load images from selected directory."""
        # Store current directory
        self.current_directory = directory

        # First, scan directory for models and update filter dropdown
        model_counts = self.scan_directory_for_models(directory)
        self.filter_panel.populate_models(model_counts)

        # Load images with current filters
        # Note: emit_filters will be called automatically when dropdown is updated
        # But we also need to load images initially
        self.image_grid.load_images(directory, "__ALL_IMAGES__", "")
        self.clear_preview()

    def refresh_current_directory(self):
        """Refresh the current directory by clearing cache and rescanning."""
        if not self.current_directory:
            return

        # Remove cached data for this directory
        self.metadata_cache.remove_directory_cache(self.current_directory)

        # Reload the directory (will rescan since cache is cleared)
        self.load_directory(self.current_directory)

    def scan_directory_for_models(self, directory: str) -> dict:
        """
        Scan directory and extract model names from all images.
        Returns dict with:
          - 'model_counts': dict mapping model_name -> count
          - 'total_images': total number of image files
          - 'images_with_prompt': count of images with SD prompts
          - 'images_without_model': count of images with prompt but no model
        Uses cache to avoid re-scanning unchanged directories.
        """
        if not os.path.isdir(directory):
            return {'model_counts': {}, 'total_images': 0, 'images_with_prompt': 0, 'images_without_model': 0}

        # Supported image formats
        extensions = ['.png', '.jpg', '.jpeg', '.webp']
        image_files_set = set()  # Use set to avoid duplicates on Windows

        for ext in extensions:
            image_files_set.update(Path(directory).glob(f"*{ext}"))
            image_files_set.update(Path(directory).glob(f"*{ext.upper()}"))

        # Convert back to list
        image_files = list(image_files_set)
        total_images = len(image_files)

        # Check cache first
        if self.metadata_cache.is_cache_valid(directory, total_images):
            print(f"Using cached metadata for {directory}")
            cached_data = self.metadata_cache.get_directory_cache(directory)

            # Check if we have the new model_stats format
            if 'model_stats' in cached_data:
                return cached_data['model_stats']

            # Old cache format - reconstruct statistics from cached metadata
            # This ensures backwards compatibility
            model_counts = cached_data.get('model_counts', {})
            images_metadata = cached_data.get('images', {})

            # Recalculate statistics from cached metadata
            images_with_prompt = 0
            images_without_model = 0

            for img_meta_dict in images_metadata.values():
                if img_meta_dict and img_meta_dict.get('positive_prompt'):
                    images_with_prompt += 1
                    if not img_meta_dict.get('model_name'):
                        images_without_model += 1

            # Return reconstructed statistics
            reconstructed_stats = {
                'model_counts': model_counts,
                'total_images': total_images,
                'images_with_prompt': images_with_prompt,
                'images_without_model': images_without_model
            }

            # Update cache with new format for next time
            self.metadata_cache.set_directory_cache(directory, model_counts, images_metadata, model_stats=reconstructed_stats)

            return reconstructed_stats

        # If no images in directory, return empty result immediately
        if total_images == 0:
            print(f"No images found in {directory}")
            # Cache the empty result
            empty_stats = {'model_counts': {}, 'total_images': 0, 'images_with_prompt': 0, 'images_without_model': 0}
            self.metadata_cache.set_directory_cache(directory, {}, {}, model_stats=empty_stats)
            return empty_stats

        # Cache miss or invalid - scan directory with progress bar
        print(f"Scanning directory {directory} ({total_images} images)")

        # Show progress bar
        self.show_progress("Loading directory", total_images)

        # Extract models from all images and gather statistics
        models = []
        images_metadata = {}
        images_with_prompt = 0
        images_without_model = 0

        for idx, img_path in enumerate(image_files):
            # Update progress
            if idx % 10 == 0:  # Update every 10 images
                self.update_progress(idx, "Loading directory")

            # Extract metadata
            metadata = extract_metadata(str(img_path))

            # Count statistics
            if metadata and metadata.positive_prompt:
                images_with_prompt += 1
                if metadata.model_name:
                    models.append(metadata.model_name)
                else:
                    images_without_model += 1

            # Store metadata for caching
            images_metadata[str(img_path)] = self.metadata_cache.metadata_to_dict(metadata)

            # Process events to keep UI responsive
            QApplication.processEvents()

        # Update to 100%
        self.update_progress(total_images, "Loading directory")

        # Count occurrences
        model_counts = dict(Counter(models))

        # Prepare statistics
        model_stats = {
            'model_counts': model_counts,
            'total_images': total_images,
            'images_with_prompt': images_with_prompt,
            'images_without_model': images_without_model
        }

        # Save to cache
        self.metadata_cache.set_directory_cache(directory, model_counts, images_metadata, model_stats=model_stats)

        # Hide progress bar
        self.hide_progress()

        return model_stats

    def apply_filters(self, model_filter: str, prompt_filter: str):
        """Apply filters to current directory."""
        if self.image_grid.current_directory:
            self.image_grid.load_images(
                self.image_grid.current_directory,
                model_filter,
                prompt_filter
            )

    def clear_preview(self):
        """Clear image preview and metadata."""
        self.current_image_path = None
        self.image_preview.clear()
        self.image_preview.setText("Select an image to preview")
        self.metadata_panel.clear_metadata()

    def on_image_selected(self, image_path: str):
        """Handle image selection - display image and metadata."""
        self.current_image_path = image_path
        self.update_preview()

        # Display metadata
        self.metadata_panel.display_metadata(image_path)

    def update_preview(self):
        """Update the image preview with proper scaling."""
        if not self.current_image_path:
            return

        try:
            pixmap = QPixmap(self.current_image_path)
            if not pixmap.isNull():
                # Scale to fit preview area while maintaining aspect ratio
                # Use slightly smaller size to account for margins
                preview_size = self.image_preview.size()
                scaled_pixmap = pixmap.scaled(
                    preview_size.width() - 20,
                    preview_size.height() - 20,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_preview.setPixmap(scaled_pixmap)
            else:
                self.image_preview.setText(f"Error loading image:\n{Path(self.current_image_path).name}")
        except Exception as e:
            self.image_preview.setText(f"Error loading image:\n{str(e)}")
            print(f"Error loading preview for {self.current_image_path}: {e}")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

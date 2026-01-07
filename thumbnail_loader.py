"""
Async thumbnail loader for background image loading.
Uses QThreadPool with QRunnable workers to load thumbnails off the main thread.
"""
import io
from typing import Optional
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image


class ThumbnailSignals(QObject):
    """Signals for thumbnail loader to communicate with main thread."""
    finished = pyqtSignal(str, QPixmap)  # (image_path, pixmap)
    error = pyqtSignal(str, str)  # (image_path, error_message)


class ThumbnailLoader(QRunnable):
    """
    Worker for loading thumbnails in background thread.
    Handles both cached and uncached images.
    """

    def __init__(self, image_path: str, metadata_cache: Optional[object] = None):
        super().__init__()
        self.image_path = image_path
        self.metadata_cache = metadata_cache
        self.signals = ThumbnailSignals()
        self.setAutoDelete(True)

    @pyqtSlot()
    def run(self):
        """Execute thumbnail loading in background thread."""
        try:
            pixmap = None

            # Try to load from cache first
            if self.metadata_cache:
                cached_thumb = self.metadata_cache.get_cached_thumbnail(self.image_path)
                if cached_thumb:
                    # Load cached JPEG (fast)
                    pixmap = QPixmap(cached_thumb)
                    if not pixmap.isNull():
                        self.signals.finished.emit(self.image_path, pixmap)
                        return

            # Cache miss - generate thumbnail with PIL
            with Image.open(self.image_path) as img:
                # Convert to RGB if needed
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                # Create thumbnail (maintains aspect ratio)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)

                # Save to cache as JPEG
                if self.metadata_cache:
                    buffer = io.BytesIO()
                    if img.mode == "RGBA":
                        # Convert RGBA to RGB with white background for JPEG
                        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3])
                        rgb_img.save(buffer, format='JPEG', quality=85)
                    else:
                        img.save(buffer, format='JPEG', quality=85)

                    self.metadata_cache.save_thumbnail(self.image_path, buffer.getvalue())

                # Convert PIL Image to QPixmap
                if img.mode == "RGB":
                    data = img.tobytes("raw", "RGB")
                    bytes_per_line = img.width * 3
                    qimage = QImage(data, img.width, img.height, bytes_per_line,
                                  QImage.Format.Format_RGB888)
                elif img.mode == "RGBA":
                    data = img.tobytes("raw", "RGBA")
                    bytes_per_line = img.width * 4
                    qimage = QImage(data, img.width, img.height, bytes_per_line,
                                  QImage.Format.Format_RGBA8888)

                # Copy image data to ensure it survives after PIL image is closed
                qimage = qimage.copy()
                pixmap = QPixmap.fromImage(qimage)

                # Emit finished signal with result
                self.signals.finished.emit(self.image_path, pixmap)

        except Exception as e:
            # Emit error signal
            self.signals.error.emit(self.image_path, str(e))

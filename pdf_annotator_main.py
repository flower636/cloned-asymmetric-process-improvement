#!/usr/bin/env python3
"""
Enhanced PDF Annotation Canvas with Complete Resize Functionality
PDFScraper v0.01 - Enhanced version with improved performance and features
"""

#!/usr/bin/env python3
"""
Enhanced PDF Annotation Canvas with Complete Resize Functionality
PDFScraper v0.01 - Enhanced version with improved performance and features
"""

import sys
import os
import json
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import IntEnum
from datetime import datetime

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLabel, QPushButton, QFileDialog, QMessageBox,
    QComboBox, QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QColorDialog, QMenu, QMenuBar, QStatusBar, QProgressBar,
    QSplitter, QFrame, QToolBar, QGroupBox, QCheckBox
)
from PyQt6.QtCore import (
    Qt, QPoint, QRect, QRectF, QSize, pyqtSignal, QThread, QMutex,
    QTimer, QPropertyAnimation, QEasingCurve, QPointF
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QAction, QFont,
    QCursor, QPainterPath, QTransform
)
# PDF processing imports
try:
    import fitz  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not found. Install with: pip install PyMuPDF")

# Excel export
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not found. Install with: pip install pandas openpyxl")

# ML/OCR imports
try:
    import cv2
    import numpy as np

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not found. Install with: pip install opencv-python")

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Warning: pytesseract not found. Install with: pip install pytesseract")


class ResizeHandle(IntEnum):
    """Enumeration for resize handle positions."""
    NONE = -1
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    TOP = 4
    BOTTOM = 5
    LEFT = 6
    RIGHT = 7


@dataclass
class Annotation:
    """Data class for storing annotation information."""
    id: str
    keyword: str
    color: str
    normalized_coords: Tuple[float, float, float, float]  # x, y, width, height (0-1)
    page_num: int
    text_content: str = ""
    confidence: float = 0.0
    created_date: str = ""
    modified_date: str = ""
    extraction_method: str = "manual"
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Annotation':
        return cls(**data)


class AnnotationCanvas(QLabel):
    """Enhanced annotation canvas with complete resize functionality."""

    annotation_added = pyqtSignal(Annotation)
    annotation_deleted = pyqtSignal(str)
    annotation_modified = pyqtSignal(Annotation)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Drawing state
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_rect = QRect()

        # Annotations storage
        self.annotations: Dict[str, Annotation] = {}
        self.selected_annotations: List[str] = []

        # PDF display
        self.pdf_pixmap: Optional[QPixmap] = None
        self.page_size: QSize = QSize()
        self.scale_factor = 1.0

        # Enhanced interaction modes and resize state
        self.interaction_mode = "draw"  # draw, select, resize
        self.resize_handle_size = 8
        self.min_annotation_size = 20  # Minimum size for annotations

        # Resize state variables
        self.resizing_annotation = None
        self.resize_handle = ResizeHandle.NONE
        self.resize_start_pos = QPoint()
        self.resize_start_coords = None
        self.dragging_annotation = None
        self.drag_start_pos = QPoint()
        self.drag_start_coords = None

        # Keywords and colors
        self.keywords = []
        self.keyword_colors = {}
        self.current_keyword = ""

        # Enable mouse tracking for resize handles
        self.setMouseTracking(True)

        # Clipboard for copy/paste
        self.clipboard_annotations = []

        # Animation support
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animations)
        self.animation_timer.start(16)  # ~60 FPS

    def set_pdf_page(self, pixmap: QPixmap, page_size: QSize):
        """Set the PDF page to display."""
        self.pdf_pixmap = pixmap
        self.page_size = page_size
        self.update_display()

    def update_display(self):
        """Update the display scaling and refresh."""
        if not self.pdf_pixmap:
            return

        # Calculate scale factor to fit widget
        widget_size = self.size()
        pixmap_size = self.pdf_pixmap.size()

        scale_x = widget_size.width() / pixmap_size.width()
        scale_y = widget_size.height() / pixmap_size.height()
        self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up

        # Scale pixmap
        scaled_size = QSize(
            int(pixmap_size.width() * self.scale_factor),
            int(pixmap_size.height() * self.scale_factor)
        )
        scaled_pixmap = self.pdf_pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        self.update_display()

    def get_screen_rect_from_annotation(self, annotation: Annotation, offset_x: int, offset_y: int) -> QRect:
        """Convert normalized annotation coordinates to screen rectangle."""
        if not self.pixmap():
            return QRect()

        pixmap_size = self.pixmap().size()
        x = annotation.normalized_coords[0] * pixmap_size.width()
        y = annotation.normalized_coords[1] * pixmap_size.height()
        width = annotation.normalized_coords[2] * pixmap_size.width()
        height = annotation.normalized_coords[3] * pixmap_size.height()

        return QRect(int(x + offset_x), int(y + offset_y), int(width), int(height))

    def get_resize_handle_rects(self, rect: QRect) -> Dict[ResizeHandle, QRect]:
        """Get rectangles for all resize handles."""
        handle_size = self.resize_handle_size
        half_size = handle_size // 2

        handles = {
            ResizeHandle.TOP_LEFT: QRect(
                rect.left() - half_size, rect.top() - half_size,
                handle_size, handle_size
            ),
            ResizeHandle.TOP_RIGHT: QRect(
                rect.right() - half_size, rect.top() - half_size,
                handle_size, handle_size
            ),
            ResizeHandle.BOTTOM_LEFT: QRect(
                rect.left() - half_size, rect.bottom() - half_size,
                handle_size, handle_size
            ),
            ResizeHandle.BOTTOM_RIGHT: QRect(
                rect.right() - half_size, rect.bottom() - half_size,
                handle_size, handle_size
            ),
            ResizeHandle.TOP: QRect(
                rect.center().x() - half_size, rect.top() - half_size,
                handle_size, handle_size
            ),
            ResizeHandle.BOTTOM: QRect(
                rect.center().x() - half_size, rect.bottom() - half_size,
                handle_size, handle_size
            ),
            ResizeHandle.LEFT: QRect(
                rect.left() - half_size, rect.center().y() - half_size,
                handle_size, handle_size
            ),
            ResizeHandle.RIGHT: QRect(
                rect.right() - half_size, rect.center().y() - half_size,
                handle_size, handle_size
            )
        }

        return handles

    def get_resize_handle_at_point(self, pos: QPoint, annotation_id: str) -> ResizeHandle:
        """Get the resize handle at the given point for the specified annotation."""
        if annotation_id not in self.annotations or annotation_id not in self.selected_annotations:
            return ResizeHandle.NONE

        # Calculate offset for centered image
        widget_rect = self.rect()
        pixmap_rect = self.pixmap().rect() if self.pixmap() else QRect()
        offset_x = (widget_rect.width() - pixmap_rect.width()) // 2
        offset_y = (widget_rect.height() - pixmap_rect.height()) // 2

        annotation = self.annotations[annotation_id]
        screen_rect = self.get_screen_rect_from_annotation(annotation, offset_x, offset_y)
        handle_rects = self.get_resize_handle_rects(screen_rect)

        for handle, handle_rect in handle_rects.items():
            if handle_rect.contains(pos):
                return handle

        return ResizeHandle.NONE

    def get_cursor_for_handle(self, handle: ResizeHandle) -> Qt.CursorShape:
        """Get appropriate cursor for resize handle."""
        cursor_map = {
            ResizeHandle.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
            ResizeHandle.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
            ResizeHandle.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
            ResizeHandle.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
            ResizeHandle.TOP: Qt.CursorShape.SizeVerCursor,
            ResizeHandle.BOTTOM: Qt.CursorShape.SizeVerCursor,
            ResizeHandle.LEFT: Qt.CursorShape.SizeHorCursor,
            ResizeHandle.RIGHT: Qt.CursorShape.SizeHorCursor,
        }
        return cursor_map.get(handle, Qt.CursorShape.ArrowCursor)

    def resize_annotation(self, annotation_id: str, handle: ResizeHandle, current_pos: QPoint):
        """Resize annotation based on handle and mouse position."""
        if annotation_id not in self.annotations:
            return

        annotation = self.annotations[annotation_id]

        # Calculate movement delta
        delta = current_pos - self.resize_start_pos

        # Get current pixmap size for coordinate conversion
        if not self.pixmap():
            return

        pixmap_size = self.pixmap().size()

        # Convert delta to normalized coordinates
        delta_norm_x = delta.x() / pixmap_size.width()
        delta_norm_y = delta.y() / pixmap_size.height()

        # Get original normalized coordinates
        orig_x, orig_y, orig_w, orig_h = self.resize_start_coords
        new_x, new_y, new_w, new_h = orig_x, orig_y, orig_w, orig_h

        # Apply resize based on handle
        if handle == ResizeHandle.TOP_LEFT:
            new_x = orig_x + delta_norm_x
            new_y = orig_y + delta_norm_y
            new_w = orig_w - delta_norm_x
            new_h = orig_h - delta_norm_y
        elif handle == ResizeHandle.TOP_RIGHT:
            new_y = orig_y + delta_norm_y
            new_w = orig_w + delta_norm_x
            new_h = orig_h - delta_norm_y
        elif handle == ResizeHandle.BOTTOM_LEFT:
            new_x = orig_x + delta_norm_x
            new_w = orig_w - delta_norm_x
            new_h = orig_h + delta_norm_y
        elif handle == ResizeHandle.BOTTOM_RIGHT:
            new_w = orig_w + delta_norm_x
            new_h = orig_h + delta_norm_y
        elif handle == ResizeHandle.TOP:
            new_y = orig_y + delta_norm_y
            new_h = orig_h - delta_norm_y
        elif handle == ResizeHandle.BOTTOM:
            new_h = orig_h + delta_norm_y
        elif handle == ResizeHandle.LEFT:
            new_x = orig_x + delta_norm_x
            new_w = orig_w - delta_norm_x
        elif handle == ResizeHandle.RIGHT:
            new_w = orig_w + delta_norm_x

        # Enforce minimum size in normalized coordinates
        min_norm_size = self.min_annotation_size / min(pixmap_size.width(), pixmap_size.height())

        if new_w < min_norm_size:
            if handle in [ResizeHandle.TOP_LEFT, ResizeHandle.BOTTOM_LEFT, ResizeHandle.LEFT]:
                new_x = orig_x + orig_w - min_norm_size
            new_w = min_norm_size

        if new_h < min_norm_size:
            if handle in [ResizeHandle.TOP_LEFT, ResizeHandle.TOP_RIGHT, ResizeHandle.TOP]:
                new_y = orig_y + orig_h - min_norm_size
            new_h = min_norm_size

        # Enforce bounds (keep within 0-1 range)
        new_x = max(0, min(1 - new_w, new_x))
        new_y = max(0, min(1 - new_h, new_y))
        new_w = min(1 - new_x, new_w)
        new_h = min(1 - new_y, new_h)

        # Update annotation coordinates
        annotation.normalized_coords = (new_x, new_y, new_w, new_h)
        self.annotation_modified.emit(annotation)
        self.update()

    def move_annotation(self, annotation_id: str, current_pos: QPoint):
        """Move annotation to new position."""
        if annotation_id not in self.annotations:
            return

        annotation = self.annotations[annotation_id]

        # Calculate movement delta
        delta = current_pos - self.drag_start_pos

        # Get current pixmap size for coordinate conversion
        if not self.pixmap():
            return

        pixmap_size = self.pixmap().size()

        # Convert delta to normalized coordinates
        delta_norm_x = delta.x() / pixmap_size.width()
        delta_norm_y = delta.y() / pixmap_size.height()

        # Get original normalized coordinates
        orig_x, orig_y, orig_w, orig_h = self.drag_start_coords

        # Calculate new position
        new_x = orig_x + delta_norm_x
        new_y = orig_y + delta_norm_y

        # Enforce bounds
        new_x = max(0, min(1 - orig_w, new_x))
        new_y = max(0, min(1 - orig_h, new_y))

        # Update annotation coordinates
        annotation.normalized_coords = (new_x, new_y, orig_w, orig_h)
        self.annotation_modified.emit(annotation)
        self.update()

    def paintEvent(self, event):
        """Custom paint event to draw annotations."""
        super().paintEvent(event)

        if not self.pdf_pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate offset for centered image
        widget_rect = self.rect()
        pixmap_rect = self.pixmap().rect() if self.pixmap() else QRect()
        offset_x = (widget_rect.width() - pixmap_rect.width()) // 2
        offset_y = (widget_rect.height() - pixmap_rect.height()) // 2

        # Draw existing annotations
        for ann_id, annotation in self.annotations.items():
            self.draw_annotation(painter, annotation, offset_x, offset_y, ann_id in self.selected_annotations)

        # Draw current rectangle being drawn
        if self.drawing and not self.current_rect.isNull():
            pen = QPen(QColor("red"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            adjusted_rect = self.current_rect.translated(offset_x, offset_y)
            painter.drawRect(adjusted_rect)

    def draw_annotation(self, painter: QPainter, annotation: Annotation,
                        offset_x: int, offset_y: int, is_selected: bool = False):
        """Draw a single annotation rectangle with enhanced handles."""
        if not self.pixmap():
            return

        screen_rect = self.get_screen_rect_from_annotation(annotation, offset_x, offset_y)

        # Set color and pen
        color = QColor(annotation.color)
        pen_width = 3 if is_selected else 2
        pen = QPen(color, pen_width)
        painter.setPen(pen)

        # Draw rectangle with enhanced visual effects
        if is_selected:
            # Add glow effect for selected annotations
            glow_pen = QPen(color.lighter(150), pen_width + 2)
            glow_pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(glow_pen)
            painter.drawRect(screen_rect)
            painter.setPen(pen)

        painter.drawRect(screen_rect)

        # Draw enhanced selection handles if selected
        if is_selected:
            self.draw_enhanced_selection_handles(painter, screen_rect)

        # Draw label with enhanced styling
        if annotation.keyword:
            self.draw_annotation_label(painter, annotation, screen_rect, color)

    def draw_enhanced_selection_handles(self, painter: QPainter, rect: QRect):
        """Draw enhanced resize handles for selected annotation."""
        handle_rects = self.get_resize_handle_rects(rect)

        # Enhanced handle colors and styling
        handle_fill = QBrush(QColor(100, 150, 255, 200))
        handle_border = QPen(QColor(50, 100, 200), 2)
        handle_border.setStyle(Qt.PenStyle.SolidLine)

        painter.setPen(handle_border)

        for handle_rect in handle_rects.values():
            painter.fillRect(handle_rect, handle_fill)
            painter.drawRect(handle_rect)

            # Add inner highlight
            inner_rect = handle_rect.adjusted(1, 1, -1, -1)
            inner_fill = QBrush(QColor(150, 200, 255, 150))
            painter.fillRect(inner_rect, inner_fill)

    def draw_annotation_label(self, painter: QPainter, annotation: Annotation,
                              screen_rect: QRect, color: QColor):
        """Draw annotation label with enhanced styling."""
        painter.setPen(QPen(QColor("black"), 1))
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        # Enhanced background with gradient effect
        text_rect = painter.fontMetrics().boundingRect(annotation.keyword)
        label_rect = QRect(screen_rect.topLeft() - QPoint(0, text_rect.height() + 4),
                           QSize(text_rect.width() + 8, text_rect.height() + 4))

        # Gradient background
        from PyQt6.QtGui import QLinearGradient
        gradient = QLinearGradient(QPointF(label_rect.topLeft()), QPointF(label_rect.bottomLeft()))
        gradient.setColorAt(0, color.lighter(160))
        gradient.setColorAt(1, color.lighter(120))
        painter.fillRect(label_rect, QBrush(gradient))

        # Border
        painter.setPen(QPen(color.darker(150), 1))
        painter.drawRect(label_rect)

        # Text
        painter.setPen(QPen(QColor("white"), 1))
        text_pos = label_rect.topLeft() + QPoint(4, text_rect.height())
        painter.drawText(text_pos, annotation.keyword)

    def mousePressEvent(self, event):
        """Enhanced mouse press event handling."""
        if not self.pixmap():
            return

        pos = event.pos()

        # Calculate offset for centered image
        widget_rect = self.rect()
        pixmap_rect = self.pixmap().rect()
        offset_x = (widget_rect.width() - pixmap_rect.width()) // 2
        offset_y = (widget_rect.height() - pixmap_rect.height()) // 2

        # Adjust position relative to image
        adjusted_pos = pos - QPoint(offset_x, offset_y)

        if event.button() == Qt.MouseButton.LeftButton:
            # Check for resize handle first (highest priority)
            resize_handle = ResizeHandle.NONE
            resize_annotation = None

            for ann_id in self.selected_annotations:
                handle = self.get_resize_handle_at_point(pos, ann_id)
                if handle != ResizeHandle.NONE:
                    resize_handle = handle
                    resize_annotation = ann_id
                    break

            if resize_handle != ResizeHandle.NONE:
                # Start resizing
                self.resizing_annotation = resize_annotation
                self.resize_handle = resize_handle
                self.resize_start_pos = adjusted_pos
                self.resize_start_coords = self.annotations[resize_annotation].normalized_coords
                return

            # Check for annotation selection/dragging
            clicked_annotation = self.get_annotation_at_point(adjusted_pos)

            if clicked_annotation and clicked_annotation in self.selected_annotations:
                # Start dragging selected annotation
                self.dragging_annotation = clicked_annotation
                self.drag_start_pos = adjusted_pos
                self.drag_start_coords = self.annotations[clicked_annotation].normalized_coords
                return

            # Handle selection or drawing based on mode
            if self.interaction_mode == "draw":
                self.start_drawing(adjusted_pos)
            elif self.interaction_mode == "select":
                self.handle_selection(adjusted_pos, event.modifiers())
            else:
                # Default to selection behavior
                self.handle_selection(adjusted_pos, event.modifiers())

        elif event.button() == Qt.MouseButton.RightButton:
            self.handle_context_menu(adjusted_pos)

    def mouseMoveEvent(self, event):
        """Enhanced mouse move event handling."""
        if not self.pixmap():
            return

        pos = event.pos()
        widget_rect = self.rect()
        pixmap_rect = self.pixmap().rect()
        offset_x = (widget_rect.width() - pixmap_rect.width()) // 2
        offset_y = (widget_rect.height() - pixmap_rect.height()) // 2
        adjusted_pos = pos - QPoint(offset_x, offset_y)

        if self.resizing_annotation and self.resize_handle != ResizeHandle.NONE:
            # Handle resizing
            self.resize_annotation(self.resizing_annotation, self.resize_handle, adjusted_pos)
        elif self.dragging_annotation:
            # Handle dragging
            self.move_annotation(self.dragging_annotation, adjusted_pos)
        elif self.drawing:
            # Handle drawing
            self.continue_drawing(adjusted_pos)
        else:
            # Update cursor based on what's under mouse
            self.update_cursor_enhanced(pos, adjusted_pos)

    def mouseReleaseEvent(self, event):
        """Enhanced mouse release event handling."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing:
                self.finish_drawing()
            elif self.resizing_annotation:
                # Finish resizing
                self.resizing_annotation = None
                self.resize_handle = ResizeHandle.NONE
                self.resize_start_pos = QPoint()
                self.resize_start_coords = None
            elif self.dragging_annotation:
                # Finish dragging
                self.dragging_annotation = None
                self.drag_start_pos = QPoint()
                self.drag_start_coords = None

    def update_cursor_enhanced(self, screen_pos: QPoint, adjusted_pos: QPoint):
        """Enhanced cursor updating with resize handle detection."""
        # Check for resize handles first
        for ann_id in self.selected_annotations:
            handle = self.get_resize_handle_at_point(screen_pos, ann_id)
            if handle != ResizeHandle.NONE:
                self.setCursor(QCursor(self.get_cursor_for_handle(handle)))
                return

        # Check for annotations
        annotation_id = self.get_annotation_at_point(adjusted_pos)

        if annotation_id and annotation_id in self.selected_annotations:
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        elif annotation_id:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def start_drawing(self, pos: QPoint):
        """Start drawing a new annotation."""
        self.drawing = True
        self.start_point = pos
        self.current_rect = QRect(pos, pos)

    def continue_drawing(self, pos: QPoint):
        """Continue drawing the current annotation."""
        self.end_point = pos
        self.current_rect = QRect(self.start_point, self.end_point).normalized()
        self.update()

    def finish_drawing(self):
        """Finish drawing and create annotation."""
        if not self.current_keyword or self.current_rect.width() < 10 or self.current_rect.height() < 10:
            self.drawing = False
            self.current_rect = QRect()
            self.update()
            return

        # Convert to normalized coordinates
        if self.pixmap():
            pixmap_size = self.pixmap().size()
            normalized_coords = (
                self.current_rect.x() / pixmap_size.width(),
                self.current_rect.y() / pixmap_size.height(),
                self.current_rect.width() / pixmap_size.width(),
                self.current_rect.height() / pixmap_size.height()
            )

            # Create annotation
            import uuid
            from datetime import datetime

            annotation = Annotation(
                id=str(uuid.uuid4()),
                keyword=self.current_keyword,
                color=self.keyword_colors.get(self.current_keyword, "#FF0000"),
                normalized_coords=normalized_coords,
                page_num=0,  # Will be set by parent
                created_date=datetime.now().isoformat(),
                modified_date=datetime.now().isoformat()
            )

            self.annotations[annotation.id] = annotation
            self.annotation_added.emit(annotation)

        # Reset drawing state
        self.drawing = False
        self.current_rect = QRect()
        self.update()

    def handle_selection(self, pos: QPoint, modifiers):
        """Handle selection of annotations."""
        clicked_annotation = self.get_annotation_at_point(pos)

        if modifiers & Qt.KeyboardModifier.ControlModifier:
            # Multi-select
            if clicked_annotation:
                if clicked_annotation in self.selected_annotations:
                    self.selected_annotations.remove(clicked_annotation)
                else:
                    self.selected_annotations.append(clicked_annotation)
        else:
            # Single select
            self.selected_annotations.clear()
            if clicked_annotation:
                self.selected_annotations.append(clicked_annotation)

        self.update()

    def handle_context_menu(self, pos: QPoint):
        """Handle right-click context menu."""
        clicked_annotation = self.get_annotation_at_point(pos)

        if clicked_annotation:
            menu = QMenu(self)

            delete_action = menu.addAction("Delete")
            copy_action = menu.addAction("Copy")
            edit_action = menu.addAction("Edit Properties")

            action = menu.exec(self.mapToGlobal(pos))

            if action == delete_action:
                self.delete_annotation_with_confirmation(clicked_annotation)
            elif action == copy_action:
                self.copy_annotations([clicked_annotation])
            elif action == edit_action:
                self.edit_annotation_properties(clicked_annotation)

    def edit_annotation_properties(self, annotation_id: str):
        """Edit annotation properties dialog."""
        if annotation_id not in self.annotations:
            return

        annotation = self.annotations[annotation_id]

        # Create simple edit dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Annotation Properties")
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)

        # Keyword selection
        layout.addWidget(QLabel("Keyword:"))
        keyword_combo = QComboBox()
        keyword_combo.addItems(self.keywords)
        keyword_combo.setCurrentText(annotation.keyword)
        layout.addWidget(keyword_combo)

        # Color selection
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color:"))
        color_btn = QPushButton()
        color_btn.setStyleSheet(f"background-color: {annotation.color}; border: 1px solid black;")
        color_btn.setFixedSize(50, 25)

        current_color = QColor(annotation.color)

        def choose_color():
            nonlocal current_color
            color = QColorDialog.getColor(current_color, dialog)
            if color.isValid():
                current_color = color
                color_btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")

        color_btn.clicked.connect(choose_color)
        color_layout.addWidget(color_btn)
        color_layout.addStretch()
        layout.addLayout(color_layout)

        # Buttons
        from PyQt6.QtWidgets import QDialogButtonBox
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Update annotation
            annotation.keyword = keyword_combo.currentText()
            annotation.color = current_color.name()
            annotation.modified_date = datetime.now().isoformat()
            self.annotation_modified.emit(annotation)
            self.update()

    def get_annotation_at_point(self, pos: QPoint) -> Optional[str]:
        """Get annotation ID at the given point."""
        if not self.pixmap():
            return None

        pixmap_size = self.pixmap().size()

        for ann_id, annotation in self.annotations.items():
            x = annotation.normalized_coords[0] * pixmap_size.width()
            y = annotation.normalized_coords[1] * pixmap_size.height()
            width = annotation.normalized_coords[2] * pixmap_size.width()
            height = annotation.normalized_coords[3] * pixmap_size.height()

            rect = QRect(int(x), int(y), int(width), int(height))
            if rect.contains(pos):
                return ann_id

        return None

    def delete_annotation_with_confirmation(self, annotation_id: str):
        """Delete annotation with confirmation dialog."""
        reply = QMessageBox()
        reply.setIcon(QMessageBox.Icon.Question)
        reply.setText("Are you sure you want to delete this annotation?")
        reply.setWindowTitle("Delete Annotation")
        reply.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        result = reply.exec()

        if result == QMessageBox.StandardButton.Yes:
            if annotation_id in self.annotations:
                del self.annotations[annotation_id]
                if annotation_id in self.selected_annotations:
                    self.selected_annotations.remove(annotation_id)
                self.annotation_deleted.emit(annotation_id)
                self.update()

    def delete_selected_annotations(self):
        """Delete all selected annotations."""
        if not self.selected_annotations:
            return

        reply = QMessageBox.question(
            self, 'Delete Annotations',
            f'Are you sure you want to delete {len(self.selected_annotations)} annotation(s)?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            for ann_id in self.selected_annotations[:]:
                if ann_id in self.annotations:
                    del self.annotations[ann_id]
                    self.annotation_deleted.emit(ann_id)

            self.selected_annotations.clear()
            self.update()

    def copy_annotations(self, annotation_ids: List[str]):
        """Copy annotations to clipboard."""
        self.clipboard_annotations = []
        for ann_id in annotation_ids:
            if ann_id in self.annotations:
                self.clipboard_annotations.append(self.annotations[ann_id])

    def paste_annotations(self):
        """Paste annotations from clipboard."""
        if not self.clipboard_annotations:
            return

        import uuid
        from datetime import datetime

        for annotation in self.clipboard_annotations:
            # Create new annotation with offset
            new_annotation = Annotation(
                id=str(uuid.uuid4()),
                keyword=annotation.keyword,
                color=annotation.color,
                normalized_coords=(
                    min(annotation.normalized_coords[0] + 0.05, 0.9),
                    min(annotation.normalized_coords[1] + 0.05, 0.9),
                    annotation.normalized_coords[2],
                    annotation.normalized_coords[3]
                ),
                page_num=annotation.page_num,
                text_content=annotation.text_content,
                confidence=annotation.confidence,
                created_date=datetime.now().isoformat(),
                modified_date=datetime.now().isoformat(),
                extraction_method=annotation.extraction_method,
                processing_time=annotation.processing_time
            )

            self.annotations[new_annotation.id] = new_annotation
            self.annotation_added.emit(new_annotation)

        self.update()

    def duplicate_selected_annotations(self):
        """Duplicate selected annotations."""
        if self.selected_annotations:
            self.copy_annotations(self.selected_annotations)
            self.paste_annotations()

    def select_all_annotations(self):
        """Select all annotations on current page."""
        self.selected_annotations = list(self.annotations.keys())
        self.update()

    def clear_selection(self):
        """Clear current selection."""
        self.selected_annotations.clear()
        self.update()

    def update_animations(self):
        """Update any ongoing animations."""
        # Placeholder for animation updates
        # Can be used for smooth transitions, highlights, etc.
        pass

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Delete and self.selected_annotations:
            self.delete_selected_annotations()
        elif event.key() == Qt.Key.Key_C and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.selected_annotations:
                self.copy_annotations(self.selected_annotations)
        elif event.key() == Qt.Key.Key_V and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.paste_annotations()
        elif event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.select_all_annotations()
        elif event.key() == Qt.Key.Key_D and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.duplicate_selected_annotations()
        elif event.key() == Qt.Key.Key_Escape:
            self.clear_selection()

        super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom functionality."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom functionality (can be implemented if needed)
            pass
        else:
            super().wheelEvent(event)

    def get_annotation_statistics(self) -> Dict[str, Any]:
        """Get statistics about current annotations."""
        stats = {
            'total_annotations': len(self.annotations),
            'selected_annotations': len(self.selected_annotations),
            'keywords_used': set(ann.keyword for ann in self.annotations.values()),
            'avg_confidence': 0.0,
            'annotation_sizes': []
        }

        if self.annotations:
            confidences = [ann.confidence for ann in self.annotations.values() if ann.confidence > 0]
            stats['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0

            for ann in self.annotations.values():
                width = ann.normalized_coords[2]
                height = ann.normalized_coords[3]
                area = width * height
                stats['annotation_sizes'].append(area)

        return stats

    def export_annotations_to_dict(self) -> Dict[str, Any]:
        """Export all annotations to dictionary format."""
        export_data = {}
        for ann_id, annotation in self.annotations.items():
            export_data[ann_id] = annotation.to_dict()
        return export_data

    def import_annotations_from_dict(self, data: Dict[str, Any]):
        """Import annotations from dictionary format."""
        self.annotations.clear()
        self.selected_annotations.clear()

        for ann_id, ann_data in data.items():
            try:
                annotation = Annotation.from_dict(ann_data)
                self.annotations[ann_id] = annotation
            except Exception as e:
                print(f"Error importing annotation {ann_id}: {e}")

        self.update()

    def set_annotation_opacity(self, opacity: float):
        """Set opacity for annotation display (0.0 to 1.0)."""
        # This can be implemented by modifying the drawing methods
        # to use the specified opacity value
        pass

    def highlight_keyword_annotations(self, keyword: str):
        """Highlight all annotations with the specified keyword."""
        highlighted = []
        for ann_id, annotation in self.annotations.items():
            if annotation.keyword == keyword:
                highlighted.append(ann_id)

        self.selected_annotations = highlighted
        self.update()

    def get_annotations_by_keyword(self, keyword: str) -> List[str]:
        """Get list of annotation IDs for a specific keyword."""
        return [ann_id for ann_id, ann in self.annotations.items()
                if ann.keyword == keyword]

    def validate_annotations(self) -> List[str]:
        """Validate annotations and return list of issues."""
        issues = []

        for ann_id, annotation in self.annotations.items():
            # Check coordinate bounds
            x, y, w, h = annotation.normalized_coords
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                issues.append(f"Annotation {ann_id} has invalid coordinates")

            if x + w > 1 or y + h > 1:
                issues.append(f"Annotation {ann_id} extends beyond page bounds")

            # Check minimum size
            if w < 0.01 or h < 0.01:  # Less than 1% of page
                issues.append(f"Annotation {ann_id} is too small")

            # Check if keyword exists
            if annotation.keyword not in self.keywords:
                issues.append(f"Annotation {ann_id} uses unknown keyword: {annotation.keyword}")

        return issues

    def auto_arrange_annotations(self):
        """Auto-arrange overlapping annotations."""
        # Simple implementation to separate overlapping annotations
        annotations_list = list(self.annotations.values())

        for i, ann1 in enumerate(annotations_list):
            for j, ann2 in enumerate(annotations_list[i + 1:], i + 1):
                if self._annotations_overlap(ann1, ann2):
                    # Move the second annotation slightly
                    x, y, w, h = ann2.normalized_coords
                    new_x = min(x + 0.05, 1 - w)
                    new_y = min(y + 0.05, 1 - h)
                    ann2.normalized_coords = (new_x, new_y, w, h)

        self.update()

    def _annotations_overlap(self, ann1: Annotation, ann2: Annotation) -> bool:
        """Check if two annotations overlap."""
        x1, y1, w1, h1 = ann1.normalized_coords
        x2, y2, w2, h2 = ann2.normalized_coords

        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    def create_annotation_thumbnail(self, annotation_id: str, size: QSize = QSize(100, 100)) -> Optional[QPixmap]:
        """Create a thumbnail image of the annotation region."""
        if annotation_id not in self.annotations or not self.pdf_pixmap:
            return None

        annotation = self.annotations[annotation_id]

        # Extract the region from the PDF pixmap
        pixmap_size = self.pdf_pixmap.size()
        x = int(annotation.normalized_coords[0] * pixmap_size.width())
        y = int(annotation.normalized_coords[1] * pixmap_size.height())
        width = int(annotation.normalized_coords[2] * pixmap_size.width())
        height = int(annotation.normalized_coords[3] * pixmap_size.height())

        region_rect = QRect(x, y, width, height)
        region_pixmap = self.pdf_pixmap.copy(region_rect)

        # Scale to thumbnail size
        thumbnail = region_pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)

        return thumbnail

    def get_selection_bounds(self) -> Optional[QRect]:
        """Get bounding rectangle of all selected annotations."""
        if not self.selected_annotations or not self.pixmap():
            return None

        pixmap_size = self.pixmap().size()
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for ann_id in self.selected_annotations:
            if ann_id in self.annotations:
                annotation = self.annotations[ann_id]
                x, y, w, h = annotation.normalized_coords

                x_pixel = x * pixmap_size.width()
                y_pixel = y * pixmap_size.height()
                w_pixel = w * pixmap_size.width()
                h_pixel = h * pixmap_size.height()

                min_x = min(min_x, x_pixel)
                min_y = min(min_y, y_pixel)
                max_x = max(max_x, x_pixel + w_pixel)
                max_y = max(max_y, y_pixel + h_pixel)

        if min_x != float('inf'):
            return QRect(int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

        return None

# Example usage and testing
# if __name__ == "__main__":

import sys
from datetime import datetime

class TestMainWindow(QMainWindow):
    """Test window for the enhanced annotation canvas."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced PDF Annotation Canvas Test - PDFScraper v0.01")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create layout
        layout = QVBoxLayout(central_widget)

        # Create toolbar
        toolbar_layout = QHBoxLayout()

        # Mode selection
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Draw", "Select"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        toolbar_layout.addWidget(QLabel("Mode:"))
        toolbar_layout.addWidget(self.mode_combo)

        # Keyword selection
        self.keyword_combo = QComboBox()
        self.keyword_combo.addItems(["Valve", "Pump", "Tank", "Pipe", "Instrument"])
        self.keyword_combo.currentTextChanged.connect(self.on_keyword_changed)
        toolbar_layout.addWidget(QLabel("Keyword:"))
        toolbar_layout.addWidget(self.keyword_combo)

        # Action buttons
        load_btn = QPushButton("Load PDF")
        load_btn.clicked.connect(self.load_pdf)
        toolbar_layout.addWidget(load_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_annotations)
        toolbar_layout.addWidget(clear_btn)

        stats_btn = QPushButton("Show Stats")
        stats_btn.clicked.connect(self.show_statistics)
        toolbar_layout.addWidget(stats_btn)

        validate_btn = QPushButton("Validate")
        validate_btn.clicked.connect(self.validate_annotations)
        toolbar_layout.addWidget(validate_btn)

        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)

        # Create annotation canvas
        self.canvas = AnnotationCanvas()
        self.canvas.annotation_added.connect(self.on_annotation_added)
        self.canvas.annotation_deleted.connect(self.on_annotation_deleted)
        self.canvas.annotation_modified.connect(self.on_annotation_modified)

        # Set up initial keywords and colors
        self.setup_keywords()

        # Add canvas to scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Select mode and keyword, then draw or select annotations")

        # Set initial values
        self.on_mode_changed("Draw")
        self.on_keyword_changed("Valve")

    def setup_keywords(self):
        """Set up keyword colors."""
        colors = {
            "Valve": "#FF0000",  # Red
            "Pump": "#00FF00",  # Green
            "Tank": "#0000FF",  # Blue
            "Pipe": "#FF8800",  # Orange
            "Instrument": "#8800FF"  # Purple
        }

        self.canvas.keywords = list(colors.keys())
        self.canvas.keyword_colors = colors

    def on_mode_changed(self, mode_text):
        """Handle mode change."""
        mode_map = {"Draw": "draw", "Select": "select"}
        self.canvas.interaction_mode = mode_map.get(mode_text, "draw")
        self.status_bar.showMessage(f"Mode: {mode_text}")

    def on_keyword_changed(self, keyword):
        """Handle keyword change."""
        self.canvas.current_keyword = keyword
        self.status_bar.showMessage(f"Current keyword: {keyword}")

    def load_pdf(self):
        """Load a PDF file for testing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", "", "PDF Files (*.pdf);;Image Files (*.png *.jpg *.jpeg)"
        )

        if file_path:
            if file_path.lower().endswith('.pdf'):
                self.load_pdf_file(file_path)
            else:
                self.load_image_file(file_path)

    def load_pdf_file(self, file_path):
        """Load PDF file."""
        if not HAS_PYMUPDF:
            QMessageBox.warning(self, "Warning", "PyMuPDF not available. Cannot load PDF.")
            return

        try:
            doc = fitz.open(file_path)
            page = doc[0]  # Load first page

            # Render page to pixmap
            mat = fitz.Matrix(2, 2)  # 2x zoom
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")

            # Convert to QPixmap
            pixmap = QPixmap()
            pixmap.loadFromData(img_data)

            self.canvas.set_pdf_page(pixmap, QSize(pix.width, pix.height))
            self.status_bar.showMessage(f"Loaded PDF: {file_path}")

            doc.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load PDF: {str(e)}")

    def load_image_file(self, file_path):
        """Load image file."""
        try:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.canvas.set_pdf_page(pixmap, pixmap.size())
                self.status_bar.showMessage(f"Loaded image: {file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to load image file.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def clear_annotations(self):
        """Clear all annotations."""
        if self.canvas.annotations:
            reply = QMessageBox.question(
                self, 'Clear Annotations',
                'Are you sure you want to clear all annotations?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.canvas.annotations.clear()
                self.canvas.selected_annotations.clear()
                self.canvas.update()
                self.status_bar.showMessage("All annotations cleared")

    def show_statistics(self):
        """Show annotation statistics."""
        stats = self.canvas.get_annotation_statistics()

        msg = f"Annotation Statistics:\n\n"
        msg += f"Total Annotations: {stats['total_annotations']}\n"
        msg += f"Selected: {stats['selected_annotations']}\n"
        msg += f"Keywords Used: {', '.join(stats['keywords_used'])}\n"
        msg += f"Average Confidence: {stats['avg_confidence']:.1f}%\n"

        if stats['annotation_sizes']:
            avg_size = sum(stats['annotation_sizes']) / len(stats['annotation_sizes'])
            msg += f"Average Size: {avg_size:.3f} (normalized)\n"

        QMessageBox.information(self, "Statistics", msg)

    def validate_annotations(self):
        """Validate annotations and show issues."""
        issues = self.canvas.validate_annotations()

        if not issues:
            QMessageBox.information(self, "Validation", "All annotations are valid!")
        else:
            msg = "Validation Issues Found:\n\n"
            msg += "\n".join(f"• {issue}" for issue in issues)
            QMessageBox.warning(self, "Validation Issues", msg)

    def on_annotation_added(self, annotation):
        """Handle annotation added."""
        self.status_bar.showMessage(
            f"Added {annotation.keyword} annotation - "
            f"Total: {len(self.canvas.annotations)}"
        )

    def on_annotation_deleted(self, annotation_id):
        """Handle annotation deleted."""
        self.status_bar.showMessage(
            f"Deleted annotation - Total: {len(self.canvas.annotations)}"
        )

    def on_annotation_modified(self, annotation):
        """Handle annotation modified."""
        coords = annotation.normalized_coords
        self.status_bar.showMessage(
            f"Modified {annotation.keyword} - "
            f"Pos: ({coords[0]:.3f}, {coords[1]:.3f}) "
            f"Size: ({coords[2]:.3f}, {coords[3]:.3f})"
        )

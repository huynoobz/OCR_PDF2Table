"""
Module 1: Input and Image Processing
Pure logic module for PDF to image conversion and image processing pipeline.
No UI code - reusable and testable.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
from typing import List, Dict, Optional, Tuple, Any
import cv2
from dataclasses import dataclass, field


@dataclass
class ImageMetadata:
    """Metadata for a processed image."""
    page_number: int
    original_size: Tuple[int, int]  # (width, height)
    processed_size: Tuple[int, int]  # (width, height)
    applied_operations: List[str] = field(default_factory=list)
    rotation: float = 0.0
    dpi: int = 300
    crop_box: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom)


@dataclass
class ProcessedImage:
    """Container for processed image and its metadata."""
    image: np.ndarray  # numpy array representation
    pil_image: Image.Image  # PIL Image for compatibility
    metadata: ImageMetadata


class PDFImageProcessor:
    """
    Main processor class for converting PDF pages to images and applying
    image processing operations.
    """
    
    def __init__(self):
        """Initialize the processor."""
        pass
    
    def process_pdf(
        self,
        pdf_path: str,
        dpi: int = 300,
        page_rotations: Optional[Dict[int, float]] = None,
        apply_grayscale: bool = True,
        apply_denoise: bool = True,
        apply_deskew: bool = True,
        apply_contrast: bool = True,
        contrast_factor: float = 1.2,
        crop_boxes: Optional[Dict[int, Tuple[int, int, int, int]]] = None
    ) -> List[ProcessedImage]:
        """
        Process a PDF file and convert each page to a processed image.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution in DPI for conversion
            page_rotations: Dict mapping page number (1-indexed) to rotation angle in degrees
            apply_grayscale: Whether to convert to grayscale
            apply_denoise: Whether to apply denoising
            apply_deskew: Whether to apply deskewing
            apply_contrast: Whether to enhance contrast
            contrast_factor: Contrast enhancement factor (1.0 = no change)
            crop_boxes: Dict mapping page number to crop box (left, top, right, bottom)
        
        Returns:
            List of ProcessedImage objects
        """
        if page_rotations is None:
            page_rotations = {}
        if crop_boxes is None:
            crop_boxes = {}
        
        # Convert PDF pages to images
        pil_images = convert_from_path(pdf_path, dpi=dpi)
        
        processed_images = []
        
        for idx, pil_img in enumerate(pil_images):
            page_num = idx + 1
            
            # Store original size
            original_size = pil_img.size
            
            # Apply rotation if specified
            rotation = page_rotations.get(page_num, 0.0)
            if rotation != 0:
                pil_img = pil_img.rotate(-rotation, expand=True)
            
            # Convert to numpy array for processing
            img_array = np.array(pil_img)
            
            # Apply processing pipeline
            operations = []
            
            if apply_grayscale:
                img_array = self._apply_grayscale(img_array)
                operations.append("grayscale")
            
            if apply_denoise:
                img_array = self._apply_denoise(img_array)
                operations.append("denoise")
            
            if apply_deskew:
                img_array, deskew_angle = self._apply_deskew(img_array)
                if deskew_angle != 0:
                    operations.append(f"deskew({deskew_angle:.2f}°)")
            
            if apply_contrast:
                img_array = self._apply_contrast(img_array, contrast_factor)
                operations.append(f"contrast({contrast_factor})")
            
            # Apply crop if specified
            crop_box = crop_boxes.get(page_num)
            if crop_box:
                img_array = self._apply_crop(img_array, crop_box)
                operations.append("crop")
            
            # Convert back to PIL Image
            if len(img_array.shape) == 2:  # Grayscale
                pil_processed = Image.fromarray(img_array, mode='L')
            elif img_array.shape[2] == 3:  # RGB
                pil_processed = Image.fromarray(img_array, mode='RGB')
            else:  # RGBA
                pil_processed = Image.fromarray(img_array, mode='RGBA')
            
            # Create metadata
            processed_size = pil_processed.size
            metadata = ImageMetadata(
                page_number=page_num,
                original_size=original_size,
                processed_size=processed_size,
                applied_operations=operations,
                rotation=rotation,
                dpi=dpi,
                crop_box=crop_box
            )
            
            processed_image = ProcessedImage(
                image=img_array,
                pil_image=pil_processed,
                metadata=metadata
            )
            
            processed_images.append(processed_image)
        
        return processed_images
    
    def _apply_grayscale(self, img_array: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                # Convert RGBA to grayscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
            else:  # RGB
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            return gray
        return img_array  # Already grayscale
    
    def _apply_denoise(self, img_array: np.ndarray) -> np.ndarray:
        """Apply denoising to the image."""
        if len(img_array.shape) == 3:
            # Color image - use fastNlMeansDenoisingColored
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        else:
            # Grayscale - use fastNlMeansDenoising
            denoised = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
        return denoised
    
    def _apply_deskew(self, img_array: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct skew in the image.
        Returns: (deskewed_image, detected_angle)
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find all non-zero points
        coords = np.column_stack(np.where(binary > 0))
        
        if len(coords) == 0:
            return img_array, 0.0
        
        # Get minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correct the angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # If angle is very small, don't rotate
        if abs(angle) < 0.5:
            return img_array, 0.0

        # Rotate without cropping (expand canvas to fit the rotated result)
        rotated = self._rotate_bound(img_array, angle)
        return rotated, angle

    def _rotate_bound(self, img_array: np.ndarray, angle_degrees: float) -> np.ndarray:
        """
        Rotate an image by angle_degrees and expand the output image to avoid cropping.

        This fixes the common OpenCV behavior where warpAffine((w,h)) clips corners.
        """
        (h, w) = img_array.shape[:2]
        center = (w / 2.0, h / 2.0)

        M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])

        # compute the new bounding dimensions of the image
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (new_w / 2.0) - center[0]
        M[1, 2] += (new_h / 2.0) - center[1]

        rotated = cv2.warpAffine(
            img_array,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
    
    def _apply_contrast(self, img_array: np.ndarray, factor: float) -> np.ndarray:
        """Enhance contrast of the image."""
        if factor == 1.0:
            return img_array
        
        # Convert to PIL for contrast enhancement
        if len(img_array.shape) == 2:
            pil_img = Image.fromarray(img_array, mode='L')
        else:
            pil_img = Image.fromarray(img_array, mode='RGB')
        
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(factor)
        
        return np.array(enhanced)
    
    def _apply_crop(self, img_array: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop the image to the specified box."""
        left, top, right, bottom = crop_box
        return img_array[top:bottom, left:right]
    
    def process_single_image(
        self,
        image: Image.Image,
        page_number: int = 1,
        apply_grayscale: bool = True,
        apply_denoise: bool = True,
        apply_deskew: bool = True,
        apply_contrast: bool = True,
        contrast_factor: float = 1.2,
        rotation: float = 0.0,
        crop_box: Optional[Tuple[int, int, int, int]] = None
    ) -> ProcessedImage:
        """
        Process a single PIL Image with the same pipeline.
        Useful for processing external images in Module 2.
        """
        original_size = image.size
        
        # Apply rotation
        if rotation != 0:
            image = image.rotate(-rotation, expand=True)
        
        img_array = np.array(image)
        operations = []
        
        if apply_grayscale:
            img_array = self._apply_grayscale(img_array)
            operations.append("grayscale")
        
        if apply_denoise:
            img_array = self._apply_denoise(img_array)
            operations.append("denoise")
        
        if apply_deskew:
            img_array, deskew_angle = self._apply_deskew(img_array)
            if deskew_angle != 0:
                operations.append(f"deskew({deskew_angle:.2f}°)")
        
        if apply_contrast:
            img_array = self._apply_contrast(img_array, contrast_factor)
            operations.append(f"contrast({contrast_factor})")
        
        if crop_box:
            img_array = self._apply_crop(img_array, crop_box)
            operations.append("crop")
        
        # Convert back to PIL
        if len(img_array.shape) == 2:
            pil_processed = Image.fromarray(img_array, mode='L')
        elif img_array.shape[2] == 3:
            pil_processed = Image.fromarray(img_array, mode='RGB')
        else:
            pil_processed = Image.fromarray(img_array, mode='RGBA')
        
        processed_size = pil_processed.size
        metadata = ImageMetadata(
            page_number=page_number,
            original_size=original_size,
            processed_size=processed_size,
            applied_operations=operations,
            rotation=rotation,
            dpi=300,  # Default for external images
            crop_box=crop_box
        )
        
        return ProcessedImage(
            image=img_array,
            pil_image=pil_processed,
            metadata=metadata
        )

    @staticmethod
    def detect_table_lines(
        img_array: np.ndarray,
        *,
        adaptive_block_size: int = 15,
        adaptive_c: int = 2,
        horizontal_kernel_len: Optional[int] = None,
        vertical_kernel_len: Optional[int] = None,
        kernel_scale: int = 30,
        filter_non_table_lines: bool = True,
        min_intersections_per_component: int = 2,
        min_component_length_px: int = 50,
        intersection_dilate_px: int = 3,
    ) -> np.ndarray:
        """
        Detect table grid lines and return a binary mask (uint8: 0/255).

        Pipeline (as requested):
        - Adaptive threshold (invert)
        - Extract horizontal lines: horizontal kernel (1 x k) + morphological open
        - Extract vertical lines: vertical kernel (k x 1) + morphological open
        - Combine horizontal + vertical masks

        Args:
            img_array: Input image (grayscale or color). Values can be uint8 or convertible.
            adaptive_block_size: Odd block size for adaptive threshold.
            adaptive_c: Constant subtracted from mean in adaptive threshold.
            horizontal_kernel_len: Kernel length for horizontal lines (pixels). If None, derived from width//kernel_scale.
            vertical_kernel_len: Kernel length for vertical lines (pixels). If None, derived from height//kernel_scale.
            kernel_scale: If kernel lengths are None, k = max(10, dimension//kernel_scale).

        Returns:
            mask: uint8 binary mask (0 background, 255 lines)
        """
        if img_array is None:
            raise ValueError("img_array is None")

        # Ensure grayscale uint8
        if img_array.ndim == 3:
            if img_array.shape[2] == 4:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Ensure valid adaptive block size
        adaptive_block_size = int(adaptive_block_size)
        if adaptive_block_size < 3:
            adaptive_block_size = 3
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1

        # Adaptive threshold (invert)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            adaptive_block_size,
            int(adaptive_c),
        )

        h, w = thresh.shape[:2]
        if horizontal_kernel_len is None:
            horizontal_kernel_len = max(10, w // max(1, int(kernel_scale)))
        if vertical_kernel_len is None:
            vertical_kernel_len = max(10, h // max(1, int(kernel_scale)))

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontal_kernel_len), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(vertical_kernel_len)))

        # Extract lines via morphological open
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

        if filter_non_table_lines:
            # Keep only components that actually participate in a grid:
            # a "table" line should intersect perpendicular lines multiple times.
            k = max(1, int(intersection_dilate_px))
            inter_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            h_d = cv2.dilate(horizontal_lines, inter_kernel, iterations=1)
            v_d = cv2.dilate(vertical_lines, inter_kernel, iterations=1)
            intersections = cv2.bitwise_and(h_d, v_d)

            def _filter_by_intersections(line_mask: np.ndarray, intersections_mask: np.ndarray, *, is_horizontal: bool) -> np.ndarray:
                num, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
                out = np.zeros_like(line_mask)
                for label in range(1, num):
                    x = stats[label, cv2.CC_STAT_LEFT]
                    y = stats[label, cv2.CC_STAT_TOP]
                    w_c = stats[label, cv2.CC_STAT_WIDTH]
                    h_c = stats[label, cv2.CC_STAT_HEIGHT]
                    # Length check (avoid tiny noise)
                    length = w_c if is_horizontal else h_c
                    if length < int(min_component_length_px):
                        continue
                    comp_mask = (labels[y : y + h_c, x : x + w_c] == label).astype(np.uint8) * 255
                    inter_roi = intersections_mask[y : y + h_c, x : x + w_c]
                    inter_count = cv2.countNonZero(cv2.bitwise_and(inter_roi, comp_mask))
                    if inter_count >= int(min_intersections_per_component):
                        out[y : y + h_c, x : x + w_c] = cv2.bitwise_or(out[y : y + h_c, x : x + w_c], comp_mask)
                return out

            horizontal_lines = _filter_by_intersections(horizontal_lines, intersections, is_horizontal=True)
            vertical_lines = _filter_by_intersections(vertical_lines, intersections, is_horizontal=False)

        # Combine
        mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        return mask

    @staticmethod
    def detect_table_cells_mask(
        img_array: np.ndarray,
        *,
        adaptive_block_size: int = 15,
        adaptive_c: int = 2,
        horizontal_kernel_len: Optional[int] = None,
        vertical_kernel_len: Optional[int] = None,
        kernel_scale: int = 30,
        # filtering params (line stage)
        filter_non_table_lines: bool = True,
        min_intersections_per_component: int = 2,
        min_component_length_px: int = 50,
        intersection_dilate_px: int = 3,
        # cell stage params
        line_dilate_px: int = 3,
        min_cell_area_px: int = 200,
        max_cell_area_ratio: float = 0.90,
        min_cell_width_px: int = 10,
        min_cell_height_px: int = 10,
    ) -> np.ndarray:
        """
        Detect table cell interiors and return a binary mask (uint8: 0/255).

        Approach:
        - Detect table lines (using detect_table_lines, optionally filtered by intersections)
        - Dilate lines to close small gaps
        - Invert lines within the table bounding box to get candidate regions
        - Connected components → keep components that look like cell interiors
          (exclude background/huge component, tiny noise)
        """
        line_mask = PDFImageProcessor.detect_table_lines(
            img_array,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            horizontal_kernel_len=horizontal_kernel_len,
            vertical_kernel_len=vertical_kernel_len,
            kernel_scale=kernel_scale,
            filter_non_table_lines=filter_non_table_lines,
            min_intersections_per_component=min_intersections_per_component,
            min_component_length_px=min_component_length_px,
            intersection_dilate_px=intersection_dilate_px,
        )

        if line_mask is None or cv2.countNonZero(line_mask) == 0:
            return np.zeros(img_array.shape[:2], dtype=np.uint8)

        ys, xs = np.where(line_mask > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        # Add a small margin
        pad = 2
        h, w = line_mask.shape[:2]
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w - 1, x1 + pad)
        y1 = min(h - 1, y1 + pad)

        roi_lines = line_mask[y0 : y1 + 1, x0 : x1 + 1]
        k = max(1, int(line_dilate_px))
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        roi_lines_d = cv2.dilate(roi_lines, dil_kernel, iterations=1)

        roi_inv = cv2.bitwise_not(roi_lines_d)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(roi_inv, connectivity=8)
        roi_area = roi_inv.shape[0] * roi_inv.shape[1]
        max_area_px = int(max_cell_area_ratio * roi_area)

        cells_roi = np.zeros_like(roi_inv)
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(min_cell_area_px) or area > max_area_px:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            cw = int(stats[label, cv2.CC_STAT_WIDTH])
            ch = int(stats[label, cv2.CC_STAT_HEIGHT])
            if cw < int(min_cell_width_px) or ch < int(min_cell_height_px):
                continue
            comp = (labels == label).astype(np.uint8) * 255
            cells_roi = cv2.bitwise_or(cells_roi, comp)

        # Place back into full image mask
        cells_mask = np.zeros((h, w), dtype=np.uint8)
        cells_mask[y0 : y1 + 1, x0 : x1 + 1] = cells_roi
        return cells_mask

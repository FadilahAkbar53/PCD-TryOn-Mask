"""
Mask overlay with alpha blending and geometric alignment.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from .utils import logger


class MaskOverlay:
    """Handle mask loading, scaling, rotation, and alpha blending."""
    
    def __init__(self, mask_path: str):
        """
        Initialize mask overlay.
        
        Args:
            mask_path: Path to mask PNG file (with alpha channel)
        """
        self.mask_path = Path(mask_path)
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Load mask with alpha channel
        self.mask_original = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        
        if self.mask_original is None:
            raise ValueError(f"Failed to load mask from {mask_path}")
        
        # Ensure 4 channels (BGRA)
        if self.mask_original.shape[2] == 3:
            # Add alpha channel if missing
            alpha = np.ones((self.mask_original.shape[0], self.mask_original.shape[1], 1), 
                           dtype=self.mask_original.dtype) * 255
            self.mask_original = np.concatenate([self.mask_original, alpha], axis=2)
            logger.warning("Mask has no alpha channel, added opaque alpha")
        
        logger.info(f"Loaded mask from {mask_path}, size: {self.mask_original.shape[:2]}")
    
    def compute_mask_transform(self, face_box: Tuple[int, int, int, int],
                               scale_width: float = 1.1,
                               scale_height: float = 0.45,
                               y_offset_ratio: float = 0.5) -> Tuple[int, int, int, int]:
        """
        Compute mask size and position based on face box.
        
        Args:
            face_box: Face bounding box (x, y, w, h)
            scale_width: Mask width as ratio of face width
            scale_height: Mask height as ratio of face height
            y_offset_ratio: Y position as ratio of face height (0.5 = middle)
        
        Returns:
            (x, y, w, h) for mask placement
        """
        face_x, face_y, face_w, face_h = face_box
        
        # Compute mask size
        mask_w = int(face_w * scale_width)
        mask_h = int(face_h * scale_height)
        
        # Compute mask position (centered horizontally, offset vertically)
        mask_x = face_x + (face_w - mask_w) // 2
        mask_y = face_y + int(face_h * y_offset_ratio)
        
        return (mask_x, mask_y, mask_w, mask_h)
    
    def resize_mask(self, width: int, height: int) -> np.ndarray:
        """
        Resize mask to specified dimensions.
        
        Args:
            width: Target width
            height: Target height
        
        Returns:
            Resized mask (BGRA)
        """
        return cv2.resize(self.mask_original, (width, height), 
                         interpolation=cv2.INTER_AREA)
    
    def rotate_mask(self, mask: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate mask by given angle.
        
        Args:
            mask: Input mask (BGRA)
            angle: Rotation angle in degrees
        
        Returns:
            Rotated mask
        """
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate with transparent background
        rotated = cv2.warpAffine(mask, M, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        
        return rotated
    
    def detect_eyes(self, face_roi: np.ndarray, 
                    cascade_path: str = None) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect eyes in face ROI for rotation alignment.
        
        Args:
            face_roi: Face region image
            cascade_path: Path to eye cascade XML
        
        Returns:
            ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y)) or None
        """
        try:
            from .utils import get_cascade_path
            
            if cascade_path is None:
                cascade_path = get_cascade_path('haarcascade_eye.xml')
            
            eye_cascade = cv2.CascadeClassifier(cascade_path)
            
            if eye_cascade.empty():
                return None
            
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(eyes) < 2:
                return None
            
            # Sort eyes by x-coordinate (left to right)
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Take first two eyes
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            # Compute eye centers
            left_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
            right_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)
            
            return (left_center, right_center)
        
        except Exception as e:
            logger.debug(f"Eye detection failed: {e}")
            return None
    
    def compute_rotation_angle(self, eyes: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        """
        Compute rotation angle from eye positions.
        
        Args:
            eyes: ((left_x, left_y), (right_x, right_y))
        
        Returns:
            Rotation angle in degrees
        """
        left_eye, right_eye = eyes
        
        # Compute angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def apply_overlay(self, image: np.ndarray, 
                     face_box: Tuple[int, int, int, int],
                     face_roi: np.ndarray = None,
                     enable_rotation: bool = False) -> np.ndarray:
        """
        Apply mask overlay to image at specified face location.
        
        Args:
            image: Input image (BGR)
            face_box: Face bounding box (x, y, w, h)
            face_roi: Face ROI for eye detection (optional)
            enable_rotation: Enable rotation based on eye alignment
        
        Returns:
            Image with mask overlay
        """
        # Compute mask transform
        mask_x, mask_y, mask_w, mask_h = self.compute_mask_transform(face_box)
        
        # Resize mask
        mask_resized = self.resize_mask(mask_w, mask_h)
        
        # Optional rotation
        angle = 0
        if enable_rotation and face_roi is not None:
            eyes = self.detect_eyes(face_roi)
            if eyes is not None:
                angle = self.compute_rotation_angle(eyes)
                mask_resized = self.rotate_mask(mask_resized, angle)
                logger.debug(f"Rotated mask by {angle:.1f} degrees")
        
        # Apply alpha blending
        result = self.alpha_blend(image, mask_resized, mask_x, mask_y)
        
        return result
    
    def alpha_blend(self, background: np.ndarray, overlay: np.ndarray,
                   x: int, y: int) -> np.ndarray:
        """
        Alpha blend overlay onto background at specified position.
        
        Args:
            background: Background image (BGR)
            overlay: Overlay image with alpha (BGRA)
            x, y: Position to place overlay
        
        Returns:
            Blended image
        """
        bg_h, bg_w = background.shape[:2]
        ov_h, ov_w = overlay.shape[:2]
        
        # Compute valid region (handle boundaries)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + ov_w)
        y2 = min(bg_h, y + ov_h)
        
        # Check if overlay is completely outside
        if x2 <= x1 or y2 <= y1:
            return background
        
        # Compute overlay region
        ov_x1 = x1 - x
        ov_y1 = y1 - y
        ov_x2 = ov_x1 + (x2 - x1)
        ov_y2 = ov_y1 + (y2 - y1)
        
        # Extract regions
        bg_region = background[y1:y2, x1:x2]
        ov_region = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
        
        # Split overlay into color and alpha
        ov_bgr = ov_region[:, :, :3]
        ov_alpha = ov_region[:, :, 3:4].astype(float) / 255.0
        
        # Alpha blending
        blended = (ov_bgr * ov_alpha + bg_region * (1 - ov_alpha)).astype(np.uint8)
        
        # Create result
        result = background.copy()
        result[y1:y2, x1:x2] = blended
        
        return result
    
    def batch_overlay(self, image: np.ndarray,
                     face_boxes: list,
                     enable_rotation: bool = False) -> np.ndarray:
        """
        Apply mask overlay to multiple faces.
        
        Args:
            image: Input image
            face_boxes: List of face boxes
            enable_rotation: Enable rotation alignment
        
        Returns:
            Image with all masks applied
        """
        result = image.copy()
        
        for face_box in face_boxes:
            x, y, w, h = face_box
            face_roi = image[y:y+h, x:x+w] if enable_rotation else None
            result = self.apply_overlay(result, face_box, face_roi, enable_rotation)
        
        return result

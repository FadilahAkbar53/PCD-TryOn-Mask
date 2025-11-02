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
        
        # Calculate mask ratio and set appropriate scaling parameters
        self.mask_ratio = self.mask_original.shape[1] / self.mask_original.shape[0]  # width/height
        self.scale_width, self.scale_height, self.y_offset_ratio = self._get_optimal_scaling()
        
        # Add rotation smoothing variables
        self.last_rotation_angle = 0.0
        self.rotation_history = []
        
        logger.info(f"Loaded mask from {mask_path}, size: {self.mask_original.shape[:2]}, ratio: {self.mask_ratio:.2f}")
    
    def _get_optimal_scaling(self) -> Tuple[float, float, float]:
        """
        Get optimal scaling parameters based on mask ratio.
        
        Returns:
            (scale_width, scale_height, y_offset_ratio)
        """
        if abs(self.mask_ratio - 1.0) < 0.1:  # Square mask (1:1 ratio)
            # Parameter untuk mask persegi (1:1)
            # scale_width: 1.1 = mask 110% lebar wajah (ubah untuk memperbesar/mengecilkan lebar)
            # scale_height: 0.9 = mask 90% tinggi wajah (ubah untuk memperbesar/mengecilkan tinggi)  
            # y_offset_ratio: 0.4 = posisi mask 40% dari atas wajah (ubah untuk naik/turun)
            return (1.2, 1.2, 0.08)
        elif self.mask_ratio > 1.5:  # Wide mask (2:1 or wider)
            # Parameter untuk mask lebar (2:1)
            return (1.1, 0.45, 0.5)
        else:  # Other ratios
            # Interpolate between square and wide
            ratio_factor = (self.mask_ratio - 1.0) / 0.5
            scale_height = 0.9 - (0.45 * ratio_factor)
            y_offset = 0.4 + (0.1 * ratio_factor)
            return (1.1, scale_height, y_offset)
    
    def compute_mask_transform(self, face_box: Tuple[int, int, int, int],
                               scale_width: float = None,
                               scale_height: float = None,
                               y_offset_ratio: float = None) -> Tuple[int, int, int, int]:
        """
        Compute mask size and position based on face box.
        
        Args:
            face_box: Face bounding box (x, y, w, h)
            scale_width: Mask width as ratio of face width (auto-detected if None)
            scale_height: Mask height as ratio of face height (auto-detected if None)
            y_offset_ratio: Y position as ratio of face height (auto-detected if None)
        
        Returns:
            (x, y, w, h) for mask placement
        """
        # Use auto-detected parameters if not provided
        if scale_width is None:
            scale_width = self.scale_width
        if scale_height is None:
            scale_height = self.scale_height
        if y_offset_ratio is None:
            y_offset_ratio = self.y_offset_ratio
            
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
        Rotate mask by given angle with enhanced interpolation.
        
        Args:
            mask: Input mask (BGRA)
            angle: Rotation angle in degrees
        
        Returns:
            Rotated mask
        """
        # Force rotation even for small angles for testing
        print(f"DEBUG - Rotating mask by {angle:.1f} degrees")
        
        if abs(angle) < 0.1:  # Only skip very tiny angles
            return mask
            
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Simple rotation without resizing for now to test
        rotated = cv2.warpAffine(mask, M, (w, h), 
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        
        return rotated
    
    def detect_eyes(self, face_roi: np.ndarray, 
                    cascade_path: str = None) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect eyes in face ROI for rotation alignment with stability checks.
        
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
            # Conservative detection to avoid false positives
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            
            if len(eyes) < 1:
                return None
            
            # More strict validation for eye detection
            valid_eyes = []
            face_h, face_w = face_roi.shape[:2]
            
            for eye in eyes:
                ex, ey, ew, eh = eye
                # Eyes should be in upper half of face
                if ey < face_h * 0.6 and ew > 15 and eh > 10:
                    valid_eyes.append(eye)
            
            if len(valid_eyes) < 1:
                return None
            
            # If only one eye detected, be very conservative
            if len(valid_eyes) == 1:
                # Don't estimate second eye, just return None to disable rotation
                return None
            
            # Sort eyes by x-coordinate (left to right)
            valid_eyes = sorted(valid_eyes, key=lambda e: e[0])
            
            # Take first two eyes
            left_eye = valid_eyes[0]
            right_eye = valid_eyes[1]
            
            # Additional validation: eyes should be reasonably spaced
            eye_distance = abs(right_eye[0] - left_eye[0])
            if eye_distance < face_w * 0.15:  # Eyes too close
                return None
            
            # Compute eye centers
            left_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
            right_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)
            
            return (left_center, right_center)
        
        except Exception as e:
            logger.debug(f"Eye detection failed: {e}")
            return None
    
    def compute_rotation_angle(self, eyes: Tuple[Tuple[int, int], Tuple[int, int]], 
                              face_roi: np.ndarray = None) -> float:
        """
        Compute rotation angle from eye positions with yaw-only estimation and smoothing.
        
        Args:
            eyes: ((left_x, left_y), (right_x, right_y))
            face_roi: Face ROI for additional analysis
        
        Returns:
            Smoothed yaw rotation angle in degrees (left-right head turn only)
        """
        left_eye, right_eye = eyes
        
        # Very conservative yaw-only calculation
        if face_roi is not None:
            face_width = face_roi.shape[1]
            face_center_x = face_width // 2
            
            # Calculate eye positions relative to face center
            left_eye_pos = left_eye[0]
            right_eye_pos = right_eye[0]
            
            # More strict eye symmetry check
            eye_span = right_eye_pos - left_eye_pos
            if eye_span < face_width * 0.2:  # Eyes too close, skip rotation
                return self.smooth_rotation(0.0)
            
            # Calculate center point between eyes
            eye_center = (left_eye_pos + right_eye_pos) / 2
            
            # Calculate offset from face center
            # When head turns LEFT: eyes appear shifted RIGHT → positive offset → mask should rotate LEFT (positive)
            # When head turns RIGHT: eyes appear shifted LEFT → negative offset → mask should rotate RIGHT (negative)
            center_offset = eye_center - face_center_x
            
            # Convert to yaw angle with correct direction
            # Positive offset = positive rotation (left turn)
            # Negative offset = negative rotation (right turn)
            raw_yaw = (center_offset / face_width) * 40  # Direct correlation
            
            # Smaller dead zone to allow more rotation
            if abs(raw_yaw) < 2:  # Reduced from 5 to 2 degrees
                raw_yaw = 0
            
            # Larger range for testing
            raw_yaw = np.clip(raw_yaw, -15, 15)  # Increased from ±8 to ±15
            
            # Apply smoothing to prevent sudden changes
            smoothed_yaw = self.smooth_rotation(raw_yaw, alpha=0.5)  # More responsive for testing
            
            # Force logging for debugging
            print(f"DEBUG - Center offset: {center_offset:.2f}, Raw yaw: {raw_yaw:.1f}°, Smoothed: {smoothed_yaw:.1f}°")
            logger.debug(f"Raw yaw: {raw_yaw:.1f}°, Smoothed: {smoothed_yaw:.1f}°")
            return smoothed_yaw
        
        # No rotation if face_roi not available
        return self.smooth_rotation(0.0)
    
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
                angle = self.compute_rotation_angle(eyes, face_roi)
                print(f"DEBUG - Applying rotation: {angle:.1f} degrees")
                mask_resized = self.rotate_mask(mask_resized, angle)
                logger.debug(f"Rotated mask by {angle:.1f} degrees")
            else:
                print("DEBUG - No eyes detected, skipping rotation")
        else:
            print(f"DEBUG - Rotation disabled: enable_rotation={enable_rotation}, face_roi={'Available' if face_roi is not None else 'None'}")
        
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
    
    def smooth_rotation(self, current_angle: float, alpha: float = 0.3) -> float:
        """
        Smooth rotation angle to prevent sudden changes.
        
        Args:
            current_angle: Current rotation angle
            alpha: Smoothing factor (0.0 = no change, 1.0 = full change)
        
        Returns:
            Smoothed rotation angle
        """
        # Add to history
        self.rotation_history.append(current_angle)
        
        # Keep only recent history
        if len(self.rotation_history) > 5:
            self.rotation_history = self.rotation_history[-5:]
        
        # Apply exponential moving average
        smoothed_angle = alpha * current_angle + (1 - alpha) * self.last_rotation_angle
        
        # Update last angle
        self.last_rotation_angle = smoothed_angle
        
        return smoothed_angle

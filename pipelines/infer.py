"""
Inference pipeline for image and video processing.
"""
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from .utils import logger, Timer, non_maximum_suppression, get_cascade_path
from .features import FeaturePipeline, ORBFeatureExtractor, BoVWEncoder
from .train import SVMTrainer
from .overlay import MaskOverlay


class FaceDetector:
    """Face detection and classification pipeline."""
    
    def __init__(self, model_dir: str = 'models', 
                 cascade_path: str = None):
        """
        Initialize face detector.
        
        Args:
            model_dir: Directory containing trained models
            cascade_path: Path to Haar cascade (auto-detect if None)
        """
        self.model_dir = Path(model_dir)
        
        # Load Haar cascade
        if cascade_path is None:
            cascade_path = get_cascade_path('haarcascade_frontalface_default.xml')
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")
        
        # Load feature pipeline
        self.orb_extractor = ORBFeatureExtractor()
        self.bovw_encoder = BoVWEncoder()
        self.bovw_encoder.load(str(self.model_dir / 'codebook.pkl'))
        
        # Load SVM classifier
        self.classifier = SVMTrainer()
        self.classifier.load(str(self.model_dir))
        
        logger.info("Face detector initialized successfully")
    
    def detect_candidates(self, image: np.ndarray,
                         scale_factor: float = 1.12,
                         min_neighbors: int = 7,
                         min_size: Tuple[int, int] = (40, 40)) -> List[Tuple[int, int, int, int]]:
        """
        Detect face candidates using Haar cascade.
        
        Args:
            image: Input image (BGR)
            scale_factor: Cascade scale factor
            min_neighbors: Cascade min neighbors
            min_size: Minimum face size
        
        Returns:
            List of face boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return [tuple(face) for face in faces]
    
    def classify_roi(self, roi: np.ndarray) -> Tuple[int, float]:
        """
        Classify a single ROI as face or non-face.
        
        Args:
            roi: ROI image (BGR)
        
        Returns:
            (prediction, confidence_score)
        """
        # Extract ORB features
        descriptors = self.orb_extractor.extract(roi)
        
        # Encode as BoVW
        bovw_features = self.bovw_encoder.encode(descriptors)
        bovw_features = bovw_features.reshape(1, -1)
        
        # Classify with better confidence handling
        if hasattr(self.classifier.svm, 'predict_proba') and self.classifier.svm.probability:
            # Use probability for better confidence (0-1 range)
            probas = self.classifier.svm.predict_proba(bovw_features)[0]
            prediction = int(probas[1] > 0.5)  # Face if probability > 0.5
            confidence = probas[1]  # Probability of being face
        else:
            # Fallback to decision function
            prediction = self.classifier.predict(bovw_features)[0]
            decision = self.classifier.decision_function(bovw_features)[0]
            # Convert decision to confidence (0-1 range)
            confidence = 1.0 / (1.0 + np.exp(-decision))  # Sigmoid
        
        return int(prediction), float(confidence)
    
    def detect(self, image: np.ndarray,
               confidence_threshold: float = 0.96,
               nms_threshold: float = 0.2) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        Detect faces in image using full pipeline.
        
        Args:
            image: Input image (BGR)
            confidence_threshold: Minimum SVM decision score
            nms_threshold: IoU threshold for NMS
        
        Returns:
            (list of face boxes, list of scores)
        """
        # Step 1: Get candidates from Haar cascade
        candidates = self.detect_candidates(image)
        
        if len(candidates) == 0:
            return [], []
        
        # Step 2: Classify each candidate
        predictions = []
        scores = []
        
        for (x, y, w, h) in candidates:
            roi = image[y:y+h, x:x+w]
            pred, score = self.classify_roi(roi)
            predictions.append(pred)
            scores.append(score)
        
        # Step 3: Filter by prediction and confidence
        valid_boxes = []
        valid_scores = []
        
        for i, (box, pred, score) in enumerate(zip(candidates, predictions, scores)):
            if pred == 1 and score >= confidence_threshold:
                valid_boxes.append(box)
                valid_scores.append(score)
        
        if len(valid_boxes) == 0:
            return [], []
        
        # Step 4: Apply NMS
        keep_indices = non_maximum_suppression(valid_boxes, valid_scores, nms_threshold)
        
        final_boxes = [valid_boxes[i] for i in keep_indices]
        final_scores = [valid_scores[i] for i in keep_indices]
        
        return final_boxes, final_scores


class ImageInference:
    """Inference on static images."""
    
    def __init__(self, model_dir: str = 'models', mask_path: str = None):
        """
        Initialize image inference.
        
        Args:
            model_dir: Directory with trained models
            mask_path: Path to specific mask PNG (for backward compatibility)
        """
        self.detector = FaceDetector(model_dir)
        self.mask_overlay = MaskOverlay(mask_path) if mask_path else None
    
    def process_image(self, image_path: str, output_path: str = None,
                     apply_mask: bool = True,
                     enable_rotation: bool = False,
                     show_boxes: bool = True) -> np.ndarray:
        """
        Process a single image.
        
        Args:
            image_path: Input image path
            output_path: Output image path (optional)
            apply_mask: Apply mask overlay
            enable_rotation: Enable mask rotation
            show_boxes: Draw bounding boxes
        
        Returns:
            Processed image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.info(f"Processing image: {image_path}")
        
        # Detect faces
        with Timer("Face detection"):
            face_boxes, scores = self.detector.detect(image)
        
        logger.info(f"Detected {len(face_boxes)} faces")
        
        # Create result
        result = image.copy()
        
        # Apply masks
        if apply_mask and self.mask_overlay and len(face_boxes) > 0:
            with Timer("Mask overlay"):
                result = self.mask_overlay.batch_overlay(result, face_boxes, enable_rotation)
        
        # Draw boxes
        if show_boxes and len(face_boxes) > 0:
            from .utils import draw_boxes
            labels = [f"Face {score:.2f}" for score in scores]
            result = draw_boxes(result, face_boxes, labels, color=(0, 255, 0))
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, result)
            logger.info(f"Saved result to {output_path}")
        
        return result


class VideoInference:
    """Inference on video streams."""
    
    def __init__(self, model_dir: str = 'models', mask_dir: str = 'assets'):
        """
        Initialize video inference.
        
        Args:
            model_dir: Directory with trained models
            mask_dir: Directory containing mask files (mask1.png - mask7.png)
        """
        self.detector = FaceDetector(model_dir)
        self.mask_dir = Path(mask_dir)
        
        # Initialize face tracker for stability
        self.face_tracker = FaceTracker(max_tracking_frames=8, iou_threshold=0.4)
        
        # Load all available masks
        self.masks: Dict[int, MaskOverlay] = {}
        for i in range(1, 8):  # mask1.png to mask7.png
            mask_path = self.mask_dir / f'mask{i}.png'
            if mask_path.exists():
                try:
                    self.masks[i] = MaskOverlay(str(mask_path))
                    logger.info(f"Loaded mask{i}.png")
                except Exception as e:
                    logger.warning(f"Failed to load mask{i}.png: {e}")
        
        if not self.masks:
            logger.warning("No masks loaded. Mask overlay will be disabled.")
            self.current_mask_id = None
        else:
            # Default to mask1
            self.current_mask_id = min(self.masks.keys())
            available_masks = ", ".join([f"mask{i}" for i in sorted(self.masks.keys())])
            logger.info(f"Available masks: {available_masks}")
            logger.info(f"Default mask set to mask{self.current_mask_id}")
        
        # Start with mask disabled (for Godot compatibility)
        self._mask_enabled = False
        logger.info("Mask overlay disabled by default (waiting for user selection)")
    
    @property
    def current_mask(self) -> MaskOverlay:
        """Get current mask overlay."""
        if self.current_mask_id is not None and self.current_mask_id in self.masks:
            return self.masks[self.current_mask_id]
        return None
    
    @property
    def mask_enabled(self) -> bool:
        """Check if mask is currently enabled."""
        return hasattr(self, '_mask_enabled') and self._mask_enabled
    
    def toggle_mask(self):
        """Toggle mask on/off."""
        if not hasattr(self, '_mask_enabled'):
            self._mask_enabled = True
        self._mask_enabled = not self._mask_enabled
    
    def switch_mask(self, mask_num: int):
        """Switch to a different mask."""
        if mask_num in self.masks:
            self.current_mask_id = mask_num
            if not hasattr(self, '_mask_enabled'):
                self._mask_enabled = True
            self._mask_enabled = True
        else:
            logger.warning(f"Mask{mask_num} not available")
    
    def process_single_frame(self, frame: np.ndarray, enable_rotation: bool = False) -> np.ndarray:
        """
        Process a single frame with face detection and mask overlay.
        
        Args:
            frame: Input frame (BGR)
            enable_rotation: Enable mask rotation based on eyes
        
        Returns:
            Processed frame with mask overlay
        """
        # Flip frame horizontally to remove mirror effect (for webcam)
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        detected_faces, scores = self.detector.detect(frame)
        
        # Use face tracker for stability
        stable_faces = self.face_tracker.update(detected_faces, scores)
        
        # Create result
        result = frame.copy()
        
        # Check if mask is enabled
        if not hasattr(self, '_mask_enabled'):
            self._mask_enabled = True
        
        # Apply masks to stable faces
        if self._mask_enabled and self.current_mask and len(stable_faces) > 0:
            result = self.current_mask.batch_overlay(result, stable_faces, enable_rotation)
        
        return result
    
    def process_video(self, video_path: str = None, output_path: str = None,
                     camera_id: int = 0,
                     apply_mask: bool = True,
                     enable_rotation: bool = False,
                     show_display: bool = True,
                     max_fps: int = 30) -> None:
        """
        Process video file or webcam stream.
        
        Args:
            video_path: Input video path (None for webcam)
            output_path: Output video path (optional)
            camera_id: Camera ID if video_path is None
            apply_mask: Apply mask overlay
            enable_rotation: Enable mask rotation
            show_display: Show live display
            max_fps: Maximum FPS for processing
        """
        # Open video source
        if video_path:
            cap = cv2.VideoCapture(video_path)
            logger.info(f"Processing video: {video_path}")
        else:
            cap = cv2.VideoCapture(camera_id)
            logger.info(f"Processing webcam (camera {camera_id})")
        
        if not cap.isOpened():
            raise ValueError("Failed to open video source")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup output writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Writing output to {output_path}")
        
        # FPS calculation variables
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        fps_update_interval = 30  # Update FPS every 30 frames
        
        # Processing loop
        frame_count = 0
        mask_enabled = apply_mask
        
        logger.info("Controls:")
        logger.info("  'q' - Quit")
        logger.info("  'm' - Toggle mask")
        logger.info("  '1-7' - Switch mask")
        logger.info("  's' - Screenshot")
        
        # Setup display window with larger size
        if show_display:
            cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
            # Set window size to be larger (1.5x the original size)
            display_width = int(width * 1.5)
            display_height = int(height * 1.5)
            cv2.resizeWindow('Face Detection', display_width, display_height)
            logger.info(f"Display window size: {display_width}x{display_height}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally to remove mirror effect (for webcam only)
                if video_path is None:  # Only flip for webcam, not for video files
                    frame = cv2.flip(frame, 1)
                
                frame_count += 1
                fps_counter += 1
                
                # Calculate FPS
                if fps_counter >= fps_update_interval:
                    elapsed_time = time.time() - fps_start_time
                    current_fps = fps_counter / elapsed_time if elapsed_time > 0 else 0
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Detect faces
                detected_faces, scores = self.detector.detect(frame)
                
                # Use face tracker for stability
                stable_faces = self.face_tracker.update(detected_faces, scores)
                
                # Create result
                result = frame.copy()
                
                # Apply masks to stable faces
                if mask_enabled and self.current_mask and len(stable_faces) > 0:
                    result = self.current_mask.batch_overlay(result, stable_faces, enable_rotation)
                elif mask_enabled and not self.current_mask:
                    logger.debug(f"Mask enabled but current_mask is None. current_mask_id: {self.current_mask_id}, available masks: {list(self.masks.keys())}")
                
                # Draw info (show both detected and stable faces count)
                if mask_enabled and self.current_mask_id is not None:
                    mask_info = f"Mask{self.current_mask_id}"
                else:
                    mask_info = "OFF"
                info_text = f"Faces: {len(stable_faces)} ({len(detected_faces)}) | FPS: {current_fps:.1f} | Mask: {mask_info}"
                cv2.putText(result, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw mask switching instructions
                if len(self.masks) > 1:
                    instruction_text = "Press 1-7 to switch masks"
                    cv2.putText(result, instruction_text, (10, height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Write to output
                if writer:
                    writer.write(result)
                
                # Display
                if show_display:
                    cv2.imshow('Face Detection', result)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit requested")
                        break
                    elif key == ord('m'):
                        mask_enabled = not mask_enabled
                        mask_status = "ON" if mask_enabled else "OFF"
                        current_mask_name = f"mask{self.current_mask_id}" if self.current_mask_id else "none"
                        logger.info(f"Mask overlay: {mask_status} (current: {current_mask_name})")
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{frame_count}.jpg"
                        cv2.imwrite(screenshot_path, result)
                        logger.info(f"Saved screenshot: {screenshot_path}")
                    elif key in [ord(str(i)) for i in range(1, 8)]:
                        # Switch mask
                        mask_num = int(chr(key))
                        if mask_num in self.masks:
                            self.current_mask_id = mask_num
                            mask_enabled = True  # Enable mask when switching
                            logger.info(f"Switched to mask{mask_num} (mask enabled)")
                        else:
                            logger.warning(f"Mask{mask_num} not available")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()
            
            logger.info(f"Processed {frame_count} frames")


class FaceTracker:
    """Face tracking and smoothing for stable mask overlay."""
    
    def __init__(self, max_tracking_frames: int = 10, iou_threshold: float = 0.5):
        """
        Initialize face tracker.
        
        Args:
            max_tracking_frames: Maximum frames to track without detection
            iou_threshold: IoU threshold for face matching
        """
        self.max_tracking_frames = max_tracking_frames
        self.iou_threshold = iou_threshold
        self.tracked_faces = []  # List of tracked face info
        self.face_id_counter = 0
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def smooth_box(self, current_box: Tuple[int, int, int, int], 
                   previous_boxes: List[Tuple[int, int, int, int]], 
                   alpha: float = 0.7) -> Tuple[int, int, int, int]:
        """Smooth face box using exponential moving average."""
        if not previous_boxes:
            return current_box
        
        # Use last box for smoothing
        prev_box = previous_boxes[-1]
        
        x = int(alpha * current_box[0] + (1 - alpha) * prev_box[0])
        y = int(alpha * current_box[1] + (1 - alpha) * prev_box[1])
        w = int(alpha * current_box[2] + (1 - alpha) * prev_box[2])
        h = int(alpha * current_box[3] + (1 - alpha) * prev_box[3])
        
        return (x, y, w, h)
    
    def update(self, detected_faces: List[Tuple[int, int, int, int]], 
               scores: List[float]) -> List[Tuple[int, int, int, int]]:
        """
        Update tracked faces with new detections.
        
        Args:
            detected_faces: New face detections
            scores: Confidence scores for detections
        
        Returns:
            Stable face boxes
        """
        current_time = time.time()
        
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_detections = list(range(len(detected_faces)))
        
        for track in self.tracked_faces:
            best_match_idx = -1
            best_iou = 0
            
            for i, face_box in enumerate(detected_faces):
                if i in unmatched_detections:
                    iou = self.calculate_iou(track['predicted_box'], face_box)
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_match_idx = i
            
            if best_match_idx >= 0:
                # Update existing track
                detected_box = detected_faces[best_match_idx]
                smoothed_box = self.smooth_box(detected_box, track['boxes'])
                
                track['boxes'].append(smoothed_box)
                track['predicted_box'] = smoothed_box
                track['last_seen'] = current_time
                track['consecutive_misses'] = 0
                track['confidence'] = scores[best_match_idx] if best_match_idx < len(scores) else 0.5
                
                # Keep only recent boxes for smoothing
                if len(track['boxes']) > 5:
                    track['boxes'] = track['boxes'][-5:]
                
                matched_tracks.append(track)
                unmatched_detections.remove(best_match_idx)
            else:
                # Track not matched, predict next position
                track['consecutive_misses'] += 1
                if track['consecutive_misses'] <= self.max_tracking_frames:
                    matched_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for i in unmatched_detections:
            face_box = detected_faces[i]
            confidence = scores[i] if i < len(scores) else 0.5
            
            new_track = {
                'id': self.face_id_counter,
                'boxes': [face_box],
                'predicted_box': face_box,
                'last_seen': current_time,
                'consecutive_misses': 0,
                'confidence': confidence
            }
            matched_tracks.append(new_track)
            self.face_id_counter += 1
        
        self.tracked_faces = matched_tracks
        
        # Return stable face boxes
        stable_faces = []
        for track in self.tracked_faces:
            if track['consecutive_misses'] <= 3:  # Only return recently seen faces
                stable_faces.append(track['predicted_box'])
        
        return stable_faces

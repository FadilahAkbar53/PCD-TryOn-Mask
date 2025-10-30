"""
Inference pipeline for image and video processing.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
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
                         scale_factor: float = 1.1,
                         min_neighbors: int = 5,
                         min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
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
        
        # Classify
        prediction = self.classifier.predict(bovw_features)[0]
        score = self.classifier.decision_function(bovw_features)[0]
        
        return int(prediction), float(score)
    
    def detect(self, image: np.ndarray,
               confidence_threshold: float = 0.0,
               nms_threshold: float = 0.3) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
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
            mask_path: Path to mask PNG
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
    
    def __init__(self, model_dir: str = 'models', mask_path: str = None):
        """
        Initialize video inference.
        
        Args:
            model_dir: Directory with trained models
            mask_path: Path to mask PNG
        """
        self.detector = FaceDetector(model_dir)
        self.mask_overlay = MaskOverlay(mask_path) if mask_path else None
    
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
        
        # Processing loop
        frame_count = 0
        mask_enabled = apply_mask
        
        logger.info("Press 'q' to quit, 'm' to toggle mask, 's' to screenshot")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect faces
                face_boxes, scores = self.detector.detect(frame)
                
                # Create result
                result = frame.copy()
                
                # Apply masks
                if mask_enabled and self.mask_overlay and len(face_boxes) > 0:
                    result = self.mask_overlay.batch_overlay(result, face_boxes, enable_rotation)
                
                # Draw info
                info_text = f"Faces: {len(face_boxes)} | Frame: {frame_count} | Mask: {'ON' if mask_enabled else 'OFF'}"
                cv2.putText(result, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
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
                        logger.info(f"Mask overlay: {'ON' if mask_enabled else 'OFF'}")
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{frame_count}.jpg"
                        cv2.imwrite(screenshot_path, result)
                        logger.info(f"Saved screenshot: {screenshot_path}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()
            
            logger.info(f"Processed {frame_count} frames")

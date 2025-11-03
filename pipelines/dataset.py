"""
Enhanced dataset loading with facial landmarks and augmentation support.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import random
from sklearn.model_selection import train_test_split
from .utils import logger, compute_iou, get_cascade_path, set_seed


class FacialLandmarks:
    """Handle facial landmarks from .pts files."""
    
    @staticmethod
    def load_landmarks(pts_file: str) -> np.ndarray:
        """
        Load facial landmarks from .pts file.
        
        Args:
            pts_file: Path to .pts file
            
        Returns:
            Array of shape (68, 2) with x,y coordinates
        """
        landmarks = []
        with open(pts_file, 'r') as f:
            lines = f.readlines()
            for line in lines[3:-1]:  # Skip header and footer
                x, y = map(float, line.strip().split())
                landmarks.append([x, y])
        return np.array(landmarks)
    
    @staticmethod
    def get_face_bbox_from_landmarks(landmarks: np.ndarray, padding: float = 0.2) -> Tuple[int, int, int, int]:
        """
        Get face bounding box from landmarks with padding.
        
        Args:
            landmarks: Array of shape (68, 2) with x,y coordinates
            padding: Padding factor (0.2 = 20% padding)
            
        Returns:
            (x, y, w, h) bounding box
        """
        min_x, min_y = landmarks.min(axis=0)
        max_x, max_y = landmarks.max(axis=0)
        
        w = max_x - min_x
        h = max_y - min_y
        
        # Add padding
        pad_w = w * padding
        pad_h = h * padding
        
        x = max(0, int(min_x - pad_w))
        y = max(0, int(min_y - pad_h))
        w = int(w + 2 * pad_w)
        h = int(h + 2 * pad_h)
        
        return x, y, w, h


class ImagePreprocessing:
    """Enhanced image preprocessing and augmentation for better model performance."""
    
    @staticmethod
    def smart_resize(image: np.ndarray, target_size: tuple = (48, 48)) -> np.ndarray:
        """Smart resize with aspect ratio preservation and padding."""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize with aspect ratio preservation
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image (center the resized image)
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=np.uint8)
        else:
            padded = np.zeros((target_h, target_w), dtype=np.uint8)
        
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        if len(image.shape) == 3:
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
        return padded
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization for better contrast."""
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel only
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    @staticmethod
    def noise_reduction(image: np.ndarray) -> np.ndarray:
        """Apply noise reduction using bilateral filter."""
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            return cv2.bilateralFilter(image, 9, 75, 75)
    
    @staticmethod
    def edge_enhancement(image: np.ndarray) -> np.ndarray:
        """Enhance edges using unsharp masking."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        if len(image.shape) == 3:
            # Convert back to color
            sharpened_color = image.copy()
            sharpened_color[:,:,0] = sharpened
            sharpened_color[:,:,1] = sharpened  
            sharpened_color[:,:,2] = sharpened
            return sharpened_color
        else:
            return sharpened

class DataAugmentation:
    """Enhanced data augmentation techniques for robust model training."""
    
    @staticmethod
    def random_brightness(image: np.ndarray, factor: float = 0.2) -> np.ndarray:
        """Randomly adjust brightness."""
        if random.random() < 0.6:
            factor = random.uniform(-factor, factor)
            image = cv2.convertScaleAbs(image, alpha=1, beta=int(255 * factor))
        return image
    
    @staticmethod
    def random_contrast(image: np.ndarray, factor: float = 0.2) -> np.ndarray:
        """Randomly adjust contrast."""
        if random.random() < 0.6:
            alpha = random.uniform(1-factor, 1+factor)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return image
    
    @staticmethod
    def random_noise(image: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
        """Add random gaussian noise."""
        if random.random() < 0.4:
            noise = np.random.normal(0, noise_factor * 255, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def random_blur(image: np.ndarray) -> np.ndarray:
        """Apply random blur."""
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image
    
    @staticmethod
    def random_rotation(image: np.ndarray, max_angle: float = 10) -> np.ndarray:
        """Apply random rotation."""
        if random.random() < 0.5:
            angle = random.uniform(-max_angle, max_angle)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))
        return image
    
    @staticmethod
    def random_horizontal_flip(image: np.ndarray) -> np.ndarray:
        """Apply random horizontal flip."""
        if random.random() < 0.5:
            return cv2.flip(image, 1)
        return image
    
    @staticmethod
    def random_scale(image: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
        """Apply random scaling."""
        if random.random() < 0.4:
            scale = random.uniform(*scale_range)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            if scale > 1.0:
                # Scale up then crop
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                return scaled[start_y:start_y+h, start_x:start_x+w]
            else:
                # Scale down then pad
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if len(image.shape) == 3:
                    padded = np.zeros((h, w, image.shape[2]), dtype=np.uint8)
                else:
                    padded = np.zeros((h, w), dtype=np.uint8)
                start_y = (h - new_h) // 2
                start_x = (w - new_w) // 2
                if len(image.shape) == 3:
                    padded[start_y:start_y+new_h, start_x:start_x+new_w] = scaled
                else:
                    padded[start_y:start_y+new_h, start_x:start_x+new_w] = scaled
                return padded
        return image
    
    @classmethod
    def apply_augmentation(cls, image: np.ndarray, augment: bool = True, 
                          intensity: str = 'light') -> np.ndarray:
        """Apply SIMPLIFIED augmentation pipeline for speed."""
        if not augment:
            return image
        
        # Apply only essential augmentations for speed
        if intensity == 'light':
            # Only brightness and horizontal flip (fastest)
            image = cls.random_brightness(image, factor=0.15)
            image = cls.random_horizontal_flip(image)
        elif intensity == 'medium':
            # Add contrast (still fast)
            image = cls.random_brightness(image, factor=0.2)
            image = cls.random_contrast(image, factor=0.15)
            image = cls.random_horizontal_flip(image)
        else:  # heavy - but still simplified
            image = cls.random_brightness(image, factor=0.25)
            image = cls.random_contrast(image, factor=0.2)
            image = cls.random_horizontal_flip(image)
            # Skip expensive operations: noise, blur, rotation, scaling
            
        return image
    
    @classmethod
    def apply_preprocessing(cls, image: np.ndarray, target_size: tuple = (48, 48),
                           enhance: bool = True) -> np.ndarray:
        """Apply SIMPLIFIED preprocessing pipeline for speed."""
        # 1. Simple resize (faster than smart resize)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        if enhance:
            # 2. Simple histogram equalization only (fastest enhancement)
            if len(image.shape) == 3:
                # Convert to grayscale for simplicity
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.equalizeHist(image)
        
        return image
        image = cls.random_noise(image)
        image = cls.random_blur(image)
        image = cls.random_rotation(image)
        
        return image


class DatasetManager:
    """Enhanced dataset manager with landmarks and augmentation support."""
    
    def __init__(self, pos_dir: str, neg_dir: str, 
                 cascade_path: str = None,
                 test_size: float = 0.15,
                 val_size: float = 0.15,
                 random_state: int = 42,
                 use_landmarks: bool = False,  # Disabled - no more .pts files needed
                 augment_data: bool = True,    # ENABLED for better performance  
                 max_pos_samples: int = None,
                 max_neg_samples: int = None,
                 preprocessing_enabled: bool = True,  # NEW: Enhanced preprocessing
                 augmentation_intensity: str = 'medium'):  # NEW: Augmentation level
        """
        Initialize dataset manager with enhanced preprocessing and augmentation.
        
        Args:
            pos_dir: Directory containing positive samples (faces)
            neg_dir: Directory containing negative samples (non-faces)
            cascade_path: Path to Haar cascade XML (auto-detect if None)
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            random_state: Random seed for reproducibility
            use_landmarks: Whether to use facial landmarks (DISABLED - no .pts files)
            augment_data: Whether to apply data augmentation (ENABLED for performance)
            max_pos_samples: Maximum positive samples to use
            max_neg_samples: Maximum negative samples to use  
            preprocessing_enabled: Whether to apply enhanced preprocessing
            augmentation_intensity: Augmentation level ('light', 'medium', 'heavy')
        """
        self.pos_dir = Path(pos_dir)
        self.neg_dir = Path(neg_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.use_landmarks = use_landmarks
        self.augment_data = augment_data
        self.max_pos_samples = max_pos_samples
        self.max_neg_samples = max_neg_samples
        self.preprocessing_enabled = preprocessing_enabled
        self.augmentation_intensity = augmentation_intensity
        self.max_pos_samples = max_pos_samples
        self.max_neg_samples = max_neg_samples
        
        # Load Haar cascade for face detection
        if cascade_path is None:
            cascade_path = get_cascade_path('haarcascade_frontalface_default.xml')
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")
        
        logger.info(f"Loaded Haar cascade from {cascade_path}")
        logger.info(f"SIMPLIFIED APPROACH: landmarks={use_landmarks}, augmentation={augment_data}")
        if not use_landmarks:
            logger.info("Using simple Haar cascade detection (like your successful friend)")
        
        # Set seed for reproducibility
        set_seed(random_state)
    
    def load_images_with_landmarks(self, directory: Path, label: int) -> List[Dict]:
        """
        Load images with optional landmarks information.
        
        Args:
            directory: Path to image directory
            label: Class label (1 for face, 0 for non-face)
        
        Returns:
            List of dictionaries with image data and metadata
        """
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        images_data = []
        
        # Get all image paths first
        all_img_paths = []
        for img_path in directory.rglob('*'):
            if img_path.suffix.lower() in supported_formats:
                # Skip mirror images to avoid duplicates in initial loading
                if '_mirror' in img_path.stem:
                    continue
                all_img_paths.append(img_path)
        
        # Apply strategic sampling for better balance (like your friend's approach)
        if label == 1 and self.max_pos_samples is not None:
            # Sample positive images
            if len(all_img_paths) > self.max_pos_samples:
                random.seed(self.random_state)
                all_img_paths = random.sample(all_img_paths, self.max_pos_samples)
                logger.info(f"Sampled {self.max_pos_samples} face images from {len(all_img_paths)} available")
        elif label == 0 and self.max_neg_samples is not None:
            # PRIORITIZE EAR IMAGES - they are crucial for reducing false positives
            ear_images = [path for path in all_img_paths if 'ear' in path.name.lower()]
            non_ear_images = [path for path in all_img_paths if 'ear' not in path.name.lower()]
            
            logger.info(f"Found {len(ear_images)} ear images and {len(non_ear_images)} other non-face images")
            
            if len(all_img_paths) > self.max_neg_samples:
                # Always include ALL ear images if possible
                selected_paths = ear_images.copy()
                remaining_slots = self.max_neg_samples - len(ear_images)
                
                if remaining_slots > 0 and non_ear_images:
                    random.seed(self.random_state)
                    additional_non_ear = random.sample(non_ear_images, min(remaining_slots, len(non_ear_images)))
                    selected_paths.extend(additional_non_ear)
                
                all_img_paths = selected_paths
                logger.info(f"Prioritized sampling: {len(ear_images)} ear images + {len(selected_paths) - len(ear_images)} other non-face images")
        elif label == 0:
            # STRATEGIC: Include all ear images + balanced selection
            ear_images = [path for path in all_img_paths if 'ear' in path.name.lower()]
            non_ear_images = [path for path in all_img_paths if 'ear' not in path.name.lower()]
            
            auto_balance = min(len(all_img_paths), 1500)  # Increased to include more ear images
            if len(all_img_paths) > auto_balance:
                # Prioritize ear images
                selected_paths = ear_images.copy()
                remaining_slots = auto_balance - len(ear_images)
                
                if remaining_slots > 0 and non_ear_images:
                    random.seed(self.random_state)
                    additional_non_ear = random.sample(non_ear_images, min(remaining_slots, len(non_ear_images)))
                    selected_paths.extend(additional_non_ear)
                
                all_img_paths = selected_paths
                logger.info(f"Strategic ear-prioritized balance: {len(ear_images)} ear images + {len(selected_paths) - len(ear_images)} others")
        
        # Load selected images
        for img_path in all_img_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                data = {
                    'image': img,
                    'label': label,
                    'path': str(img_path),
                    'landmarks': None
                }
                
                # Load landmarks if available and requested
                if self.use_landmarks and label == 1:  # Only for positive samples
                    pts_path = img_path.with_suffix('.pts')
                    if pts_path.exists():
                        try:
                            landmarks = FacialLandmarks.load_landmarks(str(pts_path))
                            data['landmarks'] = landmarks
                        except Exception as e:
                            logger.warning(f"Failed to load landmarks for {img_path}: {e}")
                
                images_data.append(data)
        
        logger.info(f"Loaded {len(images_data)} images from {directory} (label={label})")
        return images_data
    
    def extract_enhanced_face_rois(self, image_data: Dict, target_size: Tuple[int, int] = (64, 64)) -> List[Dict]:
        """
        Extract face ROIs using landmarks if available, otherwise use Haar cascade.
        
        Args:
            image_data: Dictionary with image data and metadata
            target_size: Target size for ROIs
            
        Returns:
            List of ROI dictionaries
        """
        image = image_data['image']
        rois = []
        
        # SIMPLIFIED: Always use Haar cascade (no landmarks complexity)
        # This is more reliable and matches your friend's successful approach
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
            maxSize=(300, 300)  # Limit max size for consistency
        )
        
        for (x, y, w, h) in faces:
            # Extract ROI
            roi = image[y:y+h, x:x+w]
            if roi.size > 0:
                # Apply enhanced preprocessing
                if self.preprocessing_enabled:
                    roi_processed = DataAugmentation.apply_preprocessing(
                        roi, target_size=target_size, enhance=True
                    )
                else:
                    # Simple resize fallback
                    roi_processed = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
                
                rois.append({
                    'roi': roi_processed,
                    'label': image_data['label'],
                    'source': image_data['path'],
                    'bbox': (x, y, w, h),
                    'method': 'enhanced_preprocessing' if self.preprocessing_enabled else 'simple_resize'
                })
        
        return rois
    
    def generate_enhanced_negative_rois(self, image: np.ndarray, 
                                      num_samples: int = 5,
                                      target_size: Tuple[int, int] = (64, 64)) -> List[Dict]:
        """
        Generate enhanced negative ROIs with better sampling strategy.
        
        Args:
            image: Input image
            num_samples: Number of negative samples to generate
            target_size: Target size for ROIs
            
        Returns:
            List of negative ROI dictionaries
        """
        h, w = image.shape[:2]
        min_size = min(target_size)
        max_size = min(h, w) // 2
        
        # Detect faces to avoid
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_boxes = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        negative_rois = []
        max_attempts = num_samples * 20
        attempts = 0
        
        while len(negative_rois) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Ensure image is large enough for sampling
            if min(h, w) < min_size:
                continue
            
            # Random size between min_size and max_size, but not larger than image
            max_possible_size = min(max_size, min(h, w))
            if max_possible_size <= min_size:
                size = min_size
            else:
                size = np.random.randint(min_size, max_possible_size + 1)
            
            # Random position
            if w - size <= 0 or h - size <= 0:
                continue
                
            x = np.random.randint(0, max(1, w - size))
            y = np.random.randint(0, max(1, h - size))
            
            roi_box = (x, y, size, size)
            
            # Check if overlaps with any face
            overlaps = False
            for face_box in face_boxes:
                if compute_iou(roi_box, face_box) > 0.1:
                    overlaps = True
                    break
            
            if not overlaps:
                roi = image[y:y+size, x:x+size]
                if roi.size > 0:
                    # Apply enhanced preprocessing
                    if self.preprocessing_enabled:
                        roi_processed = DataAugmentation.apply_preprocessing(
                            roi, target_size=target_size, enhance=True
                        )
                    else:
                        # Simple resize fallback
                        roi_processed = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
                    
                    negative_rois.append({
                        'roi': roi_processed,
                        'label': 0,
                        'source': 'generated_negative',
                        'bbox': roi_box,
                        'method': 'enhanced_negative' if self.preprocessing_enabled else 'simple_negative'
                    })
        
        return negative_rois
    
    def create_augmented_samples(self, roi_data: Dict, num_augmentations: int = 2) -> List[Dict]:
        """
        Create augmented versions of ROI data.
        
        Args:
            roi_data: Original ROI data
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented ROI dictionaries
        """
        augmented_samples = []
        
        for i in range(num_augmentations):
            augmented_roi = DataAugmentation.apply_augmentation(
                roi_data['roi'].copy(), 
                augment=True
            )
            
            augmented_data = roi_data.copy()
            augmented_data['roi'] = augmented_roi
            augmented_data['source'] = f"{roi_data['source']}_aug_{i}"
            augmented_samples.append(augmented_data)
        
        return augmented_samples
    
    def prepare_enhanced_dataset(self, target_size: Tuple[int, int] = (64, 64),
                               augmentation_factor: int = 2) -> Dict:
        """
        Prepare ENHANCED dataset with comprehensive preprocessing and augmentation.
        
        Args:
            target_size: Target size for all ROIs (will use smart resize)
            augmentation_factor: Number of augmented versions per positive sample (default: 2)
            
        Returns:
            Dictionary with dataset splits and metadata
        """
        logger.info("Preparing ENHANCED dataset with preprocessing & augmentation...")
        logger.info(f"Preprocessing enabled: {self.preprocessing_enabled}")
        logger.info(f"Augmentation intensity: {self.augmentation_intensity}")
        
        # Load positive and negative samples
        pos_samples = self.load_images_with_landmarks(self.pos_dir, label=1)
        neg_samples = self.load_images_with_landmarks(self.neg_dir, label=0)
        
        all_rois = []
        
        # Process positive samples
        logger.info("Processing positive samples with enhanced extraction...")
        landmarks_used = 0
        
        for sample in pos_samples:
            # Extract face ROIs (simple approach)
            face_rois = self.extract_enhanced_face_rois(sample, target_size)
            all_rois.extend(face_rois)
            
            if sample['landmarks'] is not None:
                landmarks_used += 1
            
            # SKIP augmentation completely (simple approach like friend)
            # No augmented versions for faster, simpler training
        
        logger.info(f"Extracted {len([r for r in all_rois if r['label'] == 1])} positive ROIs")
        logger.info(f"Used landmarks for {landmarks_used}/{len(pos_samples)} images")
        
        # Process negative samples (strategic quality focus)
        logger.info("Processing negative samples with enhanced quality...")
        for sample in neg_samples:
            # Generate high quality negative ROIs with preprocessing
            neg_rois = self.generate_enhanced_negative_rois(
                sample['image'], 
                num_samples=1,  # One high-quality sample per image
                target_size=target_size
            )
            all_rois.extend(neg_rois)
        
        # Apply augmentation if enabled
        if self.augment_data and augmentation_factor > 0:
            logger.info(f"Applying {self.augmentation_intensity} augmentation (factor: {augmentation_factor})...")
            
            original_positive_rois = [roi for roi in all_rois if roi['label'] == 1]
            augmented_rois = []
            
            for roi_data in original_positive_rois:
                for i in range(augmentation_factor):
                    # Apply augmentation
                    augmented_roi = DataAugmentation.apply_augmentation(
                        roi_data['roi'].copy(), 
                        augment=True, 
                        intensity=self.augmentation_intensity
                    )
                    
                    # Create augmented ROI data
                    aug_roi_data = roi_data.copy()
                    aug_roi_data['roi'] = augmented_roi
                    aug_roi_data['method'] = f"augmented_{self.augmentation_intensity}_{i+1}"
                    augmented_rois.append(aug_roi_data)
            
            all_rois.extend(augmented_rois)
            logger.info(f"Added {len(augmented_rois)} augmented positive samples")
        
        logger.info(f"Generated {len([r for r in all_rois if r['label'] == 0])} negative ROIs")
        logger.info(f"Generated {len([r for r in all_rois if r['label'] == 1])} positive ROIs (including augmented)")
        logger.info(f"Total ROIs: {len(all_rois)} (enhanced preprocessing + augmentation)")
        
        # Split dataset
        random.shuffle(all_rois)
        
        # Extract ROIs and labels
        rois = [roi_data['roi'] for roi_data in all_rois]
        labels = [roi_data['label'] for roi_data in all_rois]
        
        # Train/temp split
        train_rois, temp_rois, train_labels, temp_labels = train_test_split(
            rois, labels, 
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Val/test split
        val_rois, test_rois, val_labels, test_labels = train_test_split(
            temp_rois, temp_labels,
            test_size=self.test_size / (self.test_size + self.val_size),
            random_state=self.random_state,
            stratify=temp_labels
        )
        
        # Create dataset dictionary
        dataset = {
            'train': {'rois': train_rois, 'labels': train_labels},
            'val': {'rois': val_rois, 'labels': val_labels},
            'test': {'rois': test_rois, 'labels': test_labels},
            'metadata': {
                'pos_dir': str(self.pos_dir),
                'neg_dir': str(self.neg_dir),
                'target_size': target_size,
                'total_samples': len(all_rois),
                'positive_samples': len([r for r in all_rois if r['label'] == 1]),
                'negative_samples': len([r for r in all_rois if r['label'] == 0]),
                'landmarks_used': landmarks_used,
                'augmentation_factor': augmentation_factor if self.augment_data else 0,
                'random_state': self.random_state,
                'use_landmarks': self.use_landmarks,
                'augment_data': self.augment_data
            }
        }
        
        logger.info("=" * 60)
        logger.info("ENHANCED DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples: {dataset['metadata']['total_samples']}")
        logger.info(f"Positive samples: {dataset['metadata']['positive_samples']}")
        logger.info(f"Negative samples: {dataset['metadata']['negative_samples']}")
        logger.info(f"Landmarks used: {landmarks_used}/{len(pos_samples)} images")
        logger.info(f"Train: {len(train_rois)}, Val: {len(val_rois)}, Test: {len(test_rois)}")
        logger.info(f"Target ROI size: {target_size}")
        logger.info(f"Augmentation: {'Enabled' if self.augment_data else 'Disabled'}")
        logger.info("=" * 60)
        
        return dataset
    
    def save_dataset(self, dataset: Dict, save_path: str):
        """Save dataset metadata to JSON file."""
        metadata = dataset['metadata'].copy()
        # Don't save the actual ROI data, just metadata
        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved dataset metadata to {save_path}")
    
    def get_split_data(self, dataset: Dict, split: str) -> Tuple[List, List]:
        """Get ROIs and labels for a specific split."""
        return dataset[split]['rois'], dataset[split]['labels']

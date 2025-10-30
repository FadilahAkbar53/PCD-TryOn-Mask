"""
Dataset loading, splitting, and ROI generation using Haar cascade.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
from sklearn.model_selection import train_test_split
from .utils import logger, compute_iou, get_cascade_path, set_seed


class DatasetManager:
    """Manages dataset loading, splitting, and ROI generation."""
    
    def __init__(self, pos_dir: str, neg_dir: str, 
                 cascade_path: str = None,
                 test_size: float = 0.15,
                 val_size: float = 0.15,
                 random_state: int = 42):
        """
        Initialize dataset manager.
        
        Args:
            pos_dir: Directory containing positive samples (faces)
            neg_dir: Directory containing negative samples (non-faces)
            cascade_path: Path to Haar cascade XML (auto-detect if None)
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            random_state: Random seed for reproducibility
        """
        self.pos_dir = Path(pos_dir)
        self.neg_dir = Path(neg_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Load Haar cascade for face detection
        if cascade_path is None:
            cascade_path = get_cascade_path('haarcascade_frontalface_default.xml')
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")
        
        logger.info(f"Loaded Haar cascade from {cascade_path}")
        
        # Set seed for reproducibility
        set_seed(random_state)
    
    def load_images(self, directory: Path, label: int) -> List[Tuple[np.ndarray, int, str]]:
        """
        Load all images from a directory.
        
        Args:
            directory: Path to image directory
            label: Class label (1 for face, 0 for non-face)
        
        Returns:
            List of (image, label, path) tuples
        """
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = []
        
        for img_path in directory.rglob('*'):
            if img_path.suffix.lower() in supported_formats:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append((img, label, str(img_path)))
        
        logger.info(f"Loaded {len(images)} images from {directory} (label={label})")
        return images
    
    def extract_face_rois(self, image: np.ndarray, 
                          min_size: Tuple[int, int] = (24, 24),
                          scale_factor: float = 1.1,
                          min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Extract face ROIs using Haar cascade.
        
        Args:
            image: Input image (BGR)
            min_size: Minimum face size
            scale_factor: Cascade scale factor
            min_neighbors: Cascade min neighbors
        
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return [tuple(face) for face in faces]
    
    def generate_negative_rois(self, image: np.ndarray, 
                               num_samples: int = 5,
                               min_size: int = 48,
                               max_size: int = 200) -> List[Tuple[int, int, int, int]]:
        """
        Generate random negative ROIs that don't overlap with faces.
        
        Args:
            image: Input image
            num_samples: Number of negative samples to generate
            min_size: Minimum ROI size
            max_size: Maximum ROI size
        
        Returns:
            List of negative ROI boxes
        """
        h, w = image.shape[:2]
        
        # Detect faces to avoid
        face_boxes = self.extract_face_rois(image)
        
        negative_rois = []
        max_attempts = num_samples * 10
        attempts = 0
        
        while len(negative_rois) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Random size
            size = np.random.randint(min_size, min(max_size, min(h, w)))
            
            # Random position
            x = np.random.randint(0, max(1, w - size))
            y = np.random.randint(0, max(1, h - size))
            
            roi = (x, y, size, size)
            
            # Check if overlaps with any face
            overlaps = False
            for face_box in face_boxes:
                if compute_iou(roi, face_box) > 0.1:
                    overlaps = True
                    break
            
            if not overlaps:
                negative_rois.append(roi)
        
        return negative_rois
    
    def prepare_dataset(self, auto_generate_negatives: bool = True) -> Dict:
        """
        Prepare full dataset with train/val/test splits.
        
        Args:
            auto_generate_negatives: Generate negative ROIs from positive images
        
        Returns:
            Dictionary with dataset splits and metadata
        """
        logger.info("Preparing dataset...")
        
        # Load positive samples
        pos_samples = self.load_images(self.pos_dir, label=1)
        
        # Load negative samples
        neg_samples = self.load_images(self.neg_dir, label=0)
        
        # Generate ROIs
        all_rois = []
        
        # Process positive samples (extract faces)
        logger.info("Extracting face ROIs from positive samples...")
        for img, label, path in pos_samples:
            faces = self.extract_face_rois(img)
            for (x, y, w, h) in faces:
                roi = img[y:y+h, x:x+w]
                if roi.size > 0:
                    all_rois.append({
                        'roi': roi,
                        'label': 1,
                        'source': path,
                        'bbox': (x, y, w, h)
                    })
        
        logger.info(f"Extracted {len(all_rois)} face ROIs")
        
        # Process negative samples
        logger.info("Processing negative samples...")
        neg_count = 0
        for img, label, path in neg_samples:
            # For full negative images, generate random patches
            neg_rois = self.generate_negative_rois(img, num_samples=3)
            for (x, y, w, h) in neg_rois:
                roi = img[y:y+h, x:x+w]
                if roi.size > 0:
                    all_rois.append({
                        'roi': roi,
                        'label': 0,
                        'source': path,
                        'bbox': (x, y, w, h)
                    })
                    neg_count += 1
        
        logger.info(f"Extracted {neg_count} non-face ROIs")
        
        # Auto-generate additional negatives from positive images
        if auto_generate_negatives:
            logger.info("Auto-generating negative samples from positive images...")
            auto_neg_count = 0
            for img, _, path in pos_samples:
                neg_rois = self.generate_negative_rois(img, num_samples=2)
                for (x, y, w, h) in neg_rois:
                    roi = img[y:y+h, x:x+w]
                    if roi.size > 0:
                        all_rois.append({
                            'roi': roi,
                            'label': 0,
                            'source': path + '_auto_neg',
                            'bbox': (x, y, w, h)
                        })
                        auto_neg_count += 1
            logger.info(f"Auto-generated {auto_neg_count} additional negatives")
        
        # Shuffle
        np.random.shuffle(all_rois)
        
        # Extract labels
        labels = np.array([r['label'] for r in all_rois])
        
        # Count distribution
        pos_count = np.sum(labels == 1)
        neg_count = np.sum(labels == 0)
        logger.info(f"Dataset distribution: {pos_count} positives, {neg_count} negatives")
        
        # Split into train/val/test (stratified)
        indices = np.arange(len(all_rois))
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices, 
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_ratio = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            stratify=labels[train_val_idx],
            random_state=self.random_state
        )
        
        # Create splits
        dataset = {
            'all_rois': all_rois,
            'train_idx': train_idx.tolist(),
            'val_idx': val_idx.tolist(),
            'test_idx': test_idx.tolist(),
            'metadata': {
                'total_samples': len(all_rois),
                'positive_samples': int(pos_count),
                'negative_samples': int(neg_count),
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'test_size': len(test_idx),
                'pos_dir': str(self.pos_dir),
                'neg_dir': str(self.neg_dir),
                'random_state': self.random_state
            }
        }
        
        logger.info(f"Dataset prepared: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return dataset
    
    def save_dataset(self, dataset: Dict, save_path: str):
        """
        Save dataset splits (without ROI images, only metadata).
        
        Args:
            dataset: Dataset dictionary
            save_path: Path to save JSON
        """
        # Create lightweight version without image data
        lightweight = {
            'train_idx': dataset['train_idx'],
            'val_idx': dataset['val_idx'],
            'test_idx': dataset['test_idx'],
            'metadata': dataset['metadata'],
            'roi_info': [
                {
                    'label': int(r['label']),  # Convert numpy int to Python int
                    'source': r['source'],
                    'bbox': tuple(int(x) for x in r['bbox'])  # Convert numpy ints in bbox
                }
                for r in dataset['all_rois']
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(lightweight, f, indent=2)
        
        logger.info(f"Saved dataset splits to {save_path}")
    
    def get_split_data(self, dataset: Dict, split: str = 'train') -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get ROIs and labels for a specific split.
        
        Args:
            dataset: Dataset dictionary
            split: 'train', 'val', or 'test'
        
        Returns:
            (list of ROI images, array of labels)
        """
        idx_key = f'{split}_idx'
        if idx_key not in dataset:
            raise ValueError(f"Unknown split: {split}")
        
        indices = dataset[idx_key]
        rois = [dataset['all_rois'][i]['roi'] for i in indices]
        labels = np.array([dataset['all_rois'][i]['label'] for i in indices])
        
        return rois, labels

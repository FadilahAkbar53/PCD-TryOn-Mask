"""
ORB feature extraction and Bag of Visual Words (BoVW) encoding.
"""
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import List, Tuple
import joblib
from pathlib import Path
from .utils import logger, Timer


class ORBFeatureExtractor:
    """Extract ORB features from images."""
    
    def __init__(self, n_features: int = 500, scale_factor: float = 1.2, n_levels: int = 8):
        """
        Initialize ORB detector.
        
        Args:
            n_features: Maximum number of features to detect
            scale_factor: Pyramid decimation ratio
            n_levels: Number of pyramid levels
        """
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels
        )
        logger.info(f"Initialized ORB detector with {n_features} features")
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract ORB descriptors from an image.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            ORB descriptors (N x 32 array, uint8), or None if no keypoints
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize if too small
        if min(gray.shape[:2]) < 24:
            scale = 24 / min(gray.shape[:2])
            new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
            gray = cv2.resize(gray, new_size)
        
        # Detect and compute
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None:
            return None
        
        return descriptors
    
    def extract_batch(self, images: List[np.ndarray], verbose: bool = True) -> List[np.ndarray]:
        """
        Extract descriptors from a batch of images.
        
        Args:
            images: List of images
            verbose: Show progress
        
        Returns:
            List of descriptor arrays (some may be None)
        """
        descriptors_list = []
        
        for i, img in enumerate(images):
            desc = self.extract(img)
            descriptors_list.append(desc)
            
            if verbose and (i + 1) % 100 == 0:
                logger.info(f"Extracted features from {i + 1}/{len(images)} images")
        
        return descriptors_list


class BoVWEncoder:
    """Bag of Visual Words encoder using K-Means clustering."""
    
    def __init__(self, n_clusters: int = 256, random_state: int = 42):
        """
        Initialize BoVW encoder.
        
        Args:
            n_clusters: Number of visual words (codebook size)
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        logger.info(f"Initialized BoVW encoder with k={n_clusters}")
    
    def build_codebook(self, descriptors_list: List[np.ndarray], 
                       max_descriptors: int = 200000,
                       batch_size: int = 1024):
        """
        Build visual vocabulary using K-Means clustering.
        
        Args:
            descriptors_list: List of descriptor arrays from multiple images
            max_descriptors: Maximum number of descriptors to use
            batch_size: Batch size for MiniBatchKMeans
        """
        logger.info("Building BoVW codebook...")
        
        # Collect all descriptors
        all_descriptors = []
        for desc in descriptors_list:
            if desc is not None:
                all_descriptors.append(desc)
        
        if len(all_descriptors) == 0:
            raise ValueError("No descriptors found in dataset")
        
        # Concatenate
        all_descriptors = np.vstack(all_descriptors)
        logger.info(f"Collected {len(all_descriptors)} descriptors")
        
        # Subsample if too many
        if len(all_descriptors) > max_descriptors:
            indices = np.random.choice(len(all_descriptors), max_descriptors, replace=False)
            all_descriptors = all_descriptors[indices]
            logger.info(f"Subsampled to {max_descriptors} descriptors")
        
        # Convert to float32 for K-Means
        all_descriptors = all_descriptors.astype(np.float32)
        
        # Fit K-Means
        with Timer("K-Means clustering"):
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=batch_size,
                verbose=1,
                max_iter=100
            )
            self.kmeans.fit(all_descriptors)
        
        logger.info(f"Codebook built with {self.n_clusters} visual words")
    
    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Encode descriptors as BoVW histogram.
        
        Args:
            descriptors: ORB descriptors (N x 32)
        
        Returns:
            Normalized histogram of visual words (n_clusters,)
        """
        if self.kmeans is None:
            raise ValueError("Codebook not built yet. Call build_codebook() first.")
        
        if descriptors is None or len(descriptors) == 0:
            # Return zero histogram for images with no features
            return np.zeros(self.n_clusters, dtype=np.float32)
        
        # Convert to float32
        descriptors = descriptors.astype(np.float32)
        
        # Predict cluster assignments
        labels = self.kmeans.predict(descriptors)
        
        # Build histogram
        hist, _ = np.histogram(labels, bins=np.arange(self.n_clusters + 1))
        
        # Normalize (L2 normalization)
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        
        return hist
    
    def encode_batch(self, descriptors_list: List[np.ndarray], verbose: bool = True) -> np.ndarray:
        """
        Encode batch of descriptors.
        
        Args:
            descriptors_list: List of descriptor arrays
            verbose: Show progress
        
        Returns:
            BoVW feature matrix (n_samples x n_clusters)
        """
        features = []
        
        for i, desc in enumerate(descriptors_list):
            feat = self.encode(desc)
            features.append(feat)
            
            if verbose and (i + 1) % 100 == 0:
                logger.info(f"Encoded {i + 1}/{len(descriptors_list)} samples")
        
        return np.array(features)
    
    def save(self, path: str):
        """Save codebook to disk."""
        if self.kmeans is None:
            raise ValueError("No codebook to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.kmeans, path)
        logger.info(f"Saved codebook to {path}")
    
    def load(self, path: str):
        """Load codebook from disk."""
        self.kmeans = joblib.load(path)
        self.n_clusters = self.kmeans.n_clusters
        logger.info(f"Loaded codebook from {path} (k={self.n_clusters})")


class FeaturePipeline:
    """Complete feature extraction and encoding pipeline."""
    
    def __init__(self, orb_params: dict = None, bovw_params: dict = None):
        """
        Initialize feature pipeline.
        
        Args:
            orb_params: Parameters for ORB detector
            bovw_params: Parameters for BoVW encoder
        """
        orb_params = orb_params or {}
        bovw_params = bovw_params or {}
        
        self.orb_extractor = ORBFeatureExtractor(**orb_params)
        self.bovw_encoder = BoVWEncoder(**bovw_params)
    
    def fit(self, images: List[np.ndarray], max_descriptors: int = 200000):
        """
        Fit the pipeline on training images.
        
        Args:
            images: List of training images
            max_descriptors: Max descriptors for codebook
        """
        logger.info("Fitting feature pipeline...")
        
        # Extract all descriptors
        with Timer("ORB feature extraction"):
            descriptors_list = self.orb_extractor.extract_batch(images)
        
        # Build codebook
        self.bovw_encoder.build_codebook(descriptors_list, max_descriptors=max_descriptors)
        
        logger.info("Feature pipeline fitted successfully")
    
    def transform(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Transform images to BoVW features.
        
        Args:
            images: List of images
        
        Returns:
            Feature matrix (n_samples x n_clusters)
        """
        # Extract descriptors
        with Timer("Feature extraction"):
            descriptors_list = self.orb_extractor.extract_batch(images)
        
        # Encode as BoVW
        with Timer("BoVW encoding"):
            features = self.bovw_encoder.encode_batch(descriptors_list)
        
        return features
    
    def save_codebook(self, path: str):
        """Save codebook."""
        self.bovw_encoder.save(path)
    
    def load_codebook(self, path: str):
        """Load codebook."""
        self.bovw_encoder.load(path)

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
    """Extract ORB features from images with optimized parameters for face detection."""
    
    def __init__(self, n_features: int = 1000, scale_factor: float = 1.15, n_levels: int = 10):
        """
        Initialize ORB detector with high-quality parameters for maximum performance.
        
        Args:
            n_features: Maximum number of features to detect (1000 for high quality)
            scale_factor: Pyramid decimation ratio (1.15 for more levels)
            n_levels: Number of pyramid levels (10 for multi-scale)
        """
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=10,      # Lower threshold for more features
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,  # Better feature scoring
            patchSize=31,
            fastThreshold=8       # Lower threshold for more keypoints
        )
        logger.info(f"Initialized high-quality ORB detector: {n_features} features, scale={scale_factor}, levels={n_levels}")
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract ORB descriptors from an image with robust handling.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            ORB descriptors (N x 32 array, uint8), or None if no keypoints
        """
        if image is None or image.size == 0:
            return None
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Ensure image is large enough and well-conditioned for ORB
        h, w = gray.shape[:2]
        if min(h, w) < 20:  # Minimum size for meaningful ORB features
            scale = 32 / min(h, w)  # Scale to at least 32x32
            new_size = (int(w * scale), int(h * scale))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)
            
        # Enhance contrast for better feature detection
        gray = cv2.equalizeHist(gray)
        
        # Apply mild blur to reduce noise (helps ORB detection)
        gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
        
        # Detect and compute with error handling
        try:
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) == 0:
                # Try with more permissive ORB parameters
                orb_fallback = cv2.ORB_create(
                    nfeatures=self.orb.getMaxFeatures(),
                    scaleFactor=1.1,  # Smaller scale factor
                    nlevels=12,       # More levels
                    edgeThreshold=15, # Lower edge threshold
                    firstLevel=0,
                    WTA_K=2,
                    scoreType=cv2.ORB_HARRIS_SCORE,
                    patchSize=31,
                    fastThreshold=10  # Lower FAST threshold
                )
                keypoints, descriptors = orb_fallback.detectAndCompute(gray, None)
                
        except Exception as e:
            logger.warning(f"ORB extraction failed: {e}")
            return None
        
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
            n_clusters: Number of visual words (codebook size) - default k=256 as per prompt
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        logger.info(f"Initialized BoVW encoder with k={n_clusters} visual words")
    
    def build_codebook(self, descriptors_list: List[np.ndarray], 
                       max_descriptors: int = 200000,  # Default as per prompt requirement
                       batch_size: int = 1024):        # Larger batch for better performance
        """
        Build visual vocabulary using K-Means clustering.
        
        Args:
            descriptors_list: List of descriptor arrays from multiple images
            max_descriptors: Maximum number of descriptors to use (200k as per prompt)
            batch_size: Batch size for MiniBatchKMeans
        """
        logger.info("Building BoVW codebook...")
        
        # Collect all descriptors with debugging
        all_descriptors = []
        valid_descriptors = 0
        none_descriptors = 0
        
        for i, desc in enumerate(descriptors_list):
            if desc is not None and len(desc) > 0:
                all_descriptors.append(desc)
                valid_descriptors += 1
            else:
                none_descriptors += 1
        
        logger.info(f"Descriptor extraction results:")
        logger.info(f"  - Valid descriptors: {valid_descriptors}")
        logger.info(f"  - None/empty descriptors: {none_descriptors}")
        logger.info(f"  - Total images processed: {len(descriptors_list)}")
        
        if len(all_descriptors) == 0:
            logger.error("CRITICAL: No ORB descriptors found in any image!")
            logger.error("Possible causes:")
            logger.error("  1. Images too small/low quality")
            logger.error("  2. ORB parameters too strict")
            logger.error("  3. ROI extraction failed")
            logger.error("  4. Image format issues")
            raise ValueError("No descriptors found in dataset - check image quality and ORB parameters")
        
        # Concatenate
        all_descriptors = np.vstack(all_descriptors)
        logger.info(f"Collected {len(all_descriptors)} descriptors")
        
        # Subsample if too many (as per prompt: ≤200k)
        if len(all_descriptors) > max_descriptors:
            indices = np.random.choice(len(all_descriptors), max_descriptors, replace=False)
            all_descriptors = all_descriptors[indices]
            logger.info(f"Subsampled to {max_descriptors} descriptors for K-Means")
        
        # Convert to float32 for K-Means
        all_descriptors = all_descriptors.astype(np.float32)
        
        # Fit K-Means with standard parameters for reproducibility
        with Timer(f"K-Means clustering (k={self.n_clusters})"):
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=batch_size,
                verbose=0,
                max_iter=100,       # Standard iterations
                n_init=10,          # Multiple initializations for better clustering
                reassignment_ratio=0.01
            )
            self.kmeans.fit(all_descriptors)
        
        logger.info(f"BoVW codebook built with {self.n_clusters} visual words")
    
    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Encode ORB descriptors as normalized BoVW histogram.
        Process: ORB descriptors → K-Means assignment → histogram → L2 normalization
        
        Args:
            descriptors: ORB descriptors (N x 32 uint8)
        
        Returns:
            L2-normalized histogram of visual words (n_clusters,) as per prompt requirement
        """
        if self.kmeans is None:
            raise ValueError("Codebook not built yet. Call build_codebook() first.")
        
        if descriptors is None or len(descriptors) == 0:
            # Return zero histogram for ROIs with no ORB keypoints
            return np.zeros(self.n_clusters, dtype=np.float32)
        
        # Convert ORB descriptors to float32 for K-Means prediction
        descriptors = descriptors.astype(np.float32)
        
        # Assign each descriptor to nearest visual word (cluster center)
        labels = self.kmeans.predict(descriptors)
        
        # Build histogram of visual word assignments
        hist, _ = np.histogram(labels, bins=np.arange(self.n_clusters + 1))
        
        # L2 normalization (as per prompt: "normalized histogram of visual words")
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

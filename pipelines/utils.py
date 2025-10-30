"""
Utility functions for I/O, logging, timing, NMS, and visualization.
"""
import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Timer:
    """Simple timer context manager for performance measurement."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.name} took {self.elapsed:.3f}s")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def save_json(data: Dict[Any, Any], path: str):
    """Save dictionary to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON to {path}")


def load_json(path: str) -> Dict[Any, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {path}")
    return data


def compute_iou(box1: Tuple[int, int, int, int], 
                box2: Tuple[int, int, int, int]) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        box1, box2: (x, y, w, h) format
    
    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2) format
    box1_coords = (x1, y1, x1 + w1, y1 + h1)
    box2_coords = (x2, y2, x2 + w2, y2 + h2)
    
    # Compute intersection
    xi1 = max(box1_coords[0], box2_coords[0])
    yi1 = max(box1_coords[1], box2_coords[1])
    xi2 = min(box1_coords[2], box2_coords[2])
    yi2 = min(box1_coords[3], box2_coords[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def non_maximum_suppression(
    boxes: List[Tuple[int, int, int, int]], 
    scores: List[float], 
    iou_threshold: float = 0.3
) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        boxes: List of bounding boxes in (x, y, w, h) format
        scores: List of confidence scores for each box
        iou_threshold: IoU threshold for suppression
    
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score (descending)
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        # Keep the box with highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = [compute_iou(boxes[current], boxes[i]) for i in indices[1:]]
        
        # Keep only boxes with IoU below threshold
        indices = indices[1:][np.array(ious) < iou_threshold]
    
    return keep


def draw_boxes(
    image: np.ndarray, 
    boxes: List[Tuple[int, int, int, int]], 
    labels: List[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (BGR)
        boxes: List of boxes in (x, y, w, h) format
        labels: Optional labels for each box
        color: Box color in BGR
        thickness: Line thickness
    
    Returns:
        Image with drawn boxes
    """
    img_copy = image.copy()
    
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
        
        if labels and i < len(labels):
            label = labels[i]
            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x, y - text_h - 4), (x + text_w, y), color, -1)
            cv2.putText(img_copy, label, (x, y - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_copy


def resize_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (2x2 for binary classification)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Non-Face', 'Face']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, 
                  auc: float, save_path: str = None):
    """
    Plot Precision-Recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        auc: Area under PR curve
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AUC = {auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved PR curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, 
                   auc: float, save_path: str = None):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc: Area under ROC curve
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_cascade_path(cascade_name: str) -> str:
    """
    Get path to Haar cascade XML file.
    Try local assets first, then OpenCV installation.
    
    Args:
        cascade_name: Name of cascade (e.g., 'haarcascade_frontalface_default.xml')
    
    Returns:
        Path to cascade file
    """
    # Try local assets folder
    local_path = Path(__file__).parent.parent / "assets" / "cascades" / cascade_name
    if local_path.exists():
        return str(local_path)
    
    # Try OpenCV data folder
    cv2_data = Path(cv2.__file__).parent / "data" / cascade_name
    if cv2_data.exists():
        return str(cv2_data)
    
    # Fallback to cv2.data module
    try:
        return cv2.data.haarcascades + cascade_name
    except:
        pass
    
    raise FileNotFoundError(f"Could not find cascade: {cascade_name}")


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

"""
Main CLI application for SVM+ORB Face Detection with Mask Overlay.

Basic Usage:
    python app.py train --pos_dir data/faces --neg_dir data/non_faces
    python app.py eval --report reports/test_metrics.json
    python app.py infer --image input.jpg --out output.jpg --mask assets/mask.png
    python app.py webcam --camera 0 --mask_dir assets --show

Simplified Training (like successful friend's approach):
    python app.py train --pos_dir data/faces --neg_dir data/non_faces --max_neg 1000
    python app.py train --pos_dir data/faces --neg_dir data/non_faces --fast
    python app.py train --fast --max_pos 1000 --max_neg 500
    python app.py train --fast --svm rbf  # Try RBF kernel for better results

Webcam Controls:
    'q' - Quit
    'm' - Toggle mask on/off
    '1-9' - Switch between mask1.png to mask9.png (keyboard shortcuts)
    's' - Take screenshot
"""
import argparse
import sys
import cv2
from pathlib import Path

from pipelines.utils import logger, save_json, set_seed, ensure_dir
from pipelines.dataset import DatasetManager
from pipelines.features import FeaturePipeline
from pipelines.train import enhanced_train_pipeline, EnhancedSVMTrainer
from pipelines.infer import ImageInference, VideoInference


def train_command(args):
    """Train the face detection model with enhanced pipeline."""
    logger.info("=" * 60)
    logger.info("ENHANCED TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Prepare simplified dataset (like your successful friend's approach)
    logger.info("Step 1: Preparing SIMPLIFIED dataset...")
    
    # Apply fast mode limits if requested
    max_pos = getattr(args, 'max_pos', None)
    max_neg = getattr(args, 'max_neg', None)
    
    if getattr(args, 'fast', False):
        logger.info("🚀 FAST MODE ENABLED - Using smaller dataset for speed")
        max_pos = min(max_pos or 1500, 1500)  # Limit to 1500 faces
        max_neg = min(max_neg or 1000, 1000)  # Limit to 1000 non-faces
        logger.info(f"Fast mode limits: {max_pos} faces, {max_neg} non-faces")
    
    dataset_manager = DatasetManager(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        use_landmarks=args.use_landmarks,  # Default False for simplicity
        augment_data=args.augment_data,    # Default False for simplicity
        max_pos_samples=max_pos,
        max_neg_samples=max_neg
    )
    
    dataset = dataset_manager.prepare_enhanced_dataset(
        target_size=(args.roi_size, args.roi_size),
        augmentation_factor=args.aug_factor  # Default 0 for no augmentation
    )
    
    # Save dataset splits
    ensure_dir('data')
    dataset_manager.save_dataset(dataset, 'data/enhanced_dataset_splits.json')
    
    # Get train/val/test splits
    train_rois, train_labels = dataset_manager.get_split_data(dataset, 'train')
    val_rois, val_labels = dataset_manager.get_split_data(dataset, 'val')
    test_rois, test_labels = dataset_manager.get_split_data(dataset, 'test')
    
    logger.info(f"Train: {len(train_rois)}, Val: {len(val_rois)}, Test: {len(test_rois)}")
    
    # Step 2: Simplified feature extraction (like your successful friend)
    logger.info("Step 2: Building SIMPLIFIED feature pipeline...")
    feature_pipeline = FeaturePipeline(
        orb_params={'n_features': args.orb_features},
        bovw_params={'n_clusters': args.k, 'random_state': args.seed}
    )
    
    # Fit on training data with reduced complexity
    feature_pipeline.fit(train_rois, max_descriptors=args.max_desc)
    
    # Save codebook
    ensure_dir(args.model_dir)
    feature_pipeline.save_codebook(f'{args.model_dir}/codebook.pkl')
    
    # Transform all splits
    logger.info("Transforming training data...")
    X_train = feature_pipeline.transform(train_rois)
    
    logger.info("Transforming validation data...")
    X_val = feature_pipeline.transform(val_rois)
    
    logger.info("Transforming test data...")
    X_test = feature_pipeline.transform(test_rois)
    
    # Step 3: Train enhanced SVM classifier
    logger.info("Step 3: Training enhanced SVM classifier...")
    
    C_values = [float(c) for c in args.C.split(',')]
    gamma_values = None
    if args.svm == 'rbf' and args.gamma:
        gamma_values = args.gamma.split(',')
        # Convert to float if not 'scale' or 'auto'
        gamma_values = [g if g in ['scale', 'auto'] else float(g) for g in gamma_values]
    
    trainer, metrics = enhanced_train_pipeline(
        X_train, train_labels,
        X_val, val_labels,
        kernel=args.svm,
        C_values=C_values,
        gamma_values=gamma_values,
        model_dir=args.model_dir,
        handle_imbalance=args.handle_imbalance,
        fast_mode=getattr(args, 'fast', False)
    )
    
    # Step 4: Final evaluation on test set
    logger.info("Step 4: Final evaluation on test set...")
    test_metrics = trainer.evaluate(X_test, test_labels, save_dir='reports')
    
    # Save all metrics including dataset metadata
    all_metrics = {
        'train': metrics['train'],
        'validation': metrics['validation'],
        'test': test_metrics,
        'dataset_metadata': dataset['metadata'],
        'config': {
            'pos_dir': args.pos_dir,
            'neg_dir': args.neg_dir,
            'orb_features': args.orb_features,
            'k': args.k,
            'max_desc': args.max_desc,
            'svm_kernel': args.svm,
            'C_values': C_values,
            'gamma_values': gamma_values,
            'seed': args.seed,
            'roi_size': args.roi_size,
            'use_landmarks': args.use_landmarks,
            'augment_data': args.augment_data,
            'aug_factor': args.aug_factor,
            'handle_imbalance': args.handle_imbalance
        }
    }
    
    save_json(all_metrics, 'reports/enhanced_metrics.json')
    
    logger.info("=" * 60)
    logger.info("SIMPLIFIED TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Approach: {'WITH' if args.use_landmarks else 'WITHOUT'} landmarks (like successful friend)")
    logger.info(f"Dataset Balance: {dataset['metadata']['positive_samples']} faces vs {dataset['metadata']['negative_samples']} non-faces")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test AUC: {test_metrics['roc_auc']:.4f}")
    
    # Compare with previous results
    if test_metrics['f1'] > 0.5:
        logger.info("✅ GOOD RESULTS! F1 > 0.5")
    elif test_metrics['f1'] > 0.3:
        logger.info("⚠️  MODERATE RESULTS. F1 > 0.3 but could be better")
    else:
        logger.info("❌ POOR RESULTS. F1 < 0.3 - consider further simplification")
        
    logger.info(f"Landmarks used: {'YES' if args.use_landmarks else 'NO (simplified like friend)'}")
    logger.info(f"Augmentation: {'YES' if args.augment_data else 'NO (simplified like friend)'}")
    logger.info("=" * 60)


def eval_command(args):
    """Evaluate trained model on test set."""
    logger.info("=" * 60)
    logger.info("ENHANCED EVALUATION")
    logger.info("=" * 60)
    
    # Load dataset metadata
    from pipelines.utils import load_json
    try:
        dataset_info = load_json('data/enhanced_dataset_splits.json')
        logger.info("Using enhanced dataset metadata")
    except FileNotFoundError:
        logger.warning("Enhanced dataset metadata not found, falling back to legacy")
        dataset_info = load_json('data/dataset_splits.json')
    
    # Reconstruct dataset (simplified for evaluation)
    dataset_manager = DatasetManager(
        pos_dir=dataset_info.get('pos_dir', 'data/faces'),
        neg_dir=dataset_info.get('neg_dir', 'data/non_faces'),
        random_state=dataset_info.get('random_state', 42),
        use_landmarks=dataset_info.get('use_landmarks', True),
        augment_data=False  # No augmentation for evaluation
    )
    
    # Prepare dataset
    target_size = dataset_info.get('target_size', [64, 64])
    dataset = dataset_manager.prepare_enhanced_dataset(
        target_size=tuple(target_size),
        augmentation_factor=0
    )
    test_rois, test_labels = dataset_manager.get_split_data(dataset, 'test')
    
    # Load feature pipeline
    feature_pipeline = FeaturePipeline()
    feature_pipeline.load_codebook(f'{args.model_dir}/codebook.pkl')
    
    # Transform test data
    logger.info("Extracting features from test set...")
    X_test = feature_pipeline.transform(test_rois)
    
    # Load classifier
    classifier = EnhancedSVMTrainer()
    classifier.load(args.model_dir)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, test_labels, save_dir='reports')
    
    # Save metrics
    if args.report:
        save_json(metrics, args.report)
    
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")
    logger.info(f"AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Average Precision: {metrics.get('average_precision', 'N/A')}")
    logger.info("=" * 60)


def infer_command(args):
    """Run inference on image."""
    logger.info("=" * 60)
    logger.info("IMAGE INFERENCE")
    logger.info("=" * 60)
    
    # Check mask file
    if args.mask and not Path(args.mask).exists():
        logger.error(f"Mask file not found: {args.mask}")
        sys.exit(1)
    
    # Initialize inference
    inference = ImageInference(
        model_dir=args.model_dir,
        mask_path=args.mask
    )
    
    # Process image
    result = inference.process_image(
        image_path=args.image,
        output_path=args.out,
        apply_mask=args.mask is not None,
        enable_rotation=args.rotate,
        show_boxes=args.boxes
    )
    
    # Show result if requested
    if args.show:
        import cv2
        cv2.imshow('Result', result)
        logger.info("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETED")
    logger.info("=" * 60)


def webcam_command(args):
    """Run inference on webcam."""
    logger.info("=" * 60)
    logger.info("WEBCAM INFERENCE")
    logger.info("=" * 60)
    
    # Initialize inference with mask directory instead of single mask
    inference = VideoInference(
        model_dir=args.model_dir,
        mask_dir=args.mask_dir
    )
    
    # Process webcam
    inference.process_video(
        video_path=None,
        output_path=args.out,
        camera_id=args.camera,
        apply_mask=True,  # Always try to apply mask if available
        enable_rotation=args.rotate,
        show_display=args.show
    )
    
    logger.info("=" * 60)
    logger.info("WEBCAM PROCESSING COMPLETED")
    logger.info("=" * 60)


def list_webcams_command(args):
    """List available webcam devices."""
    logger.info("=" * 60)
    logger.info("AVAILABLE WEBCAM DEVICES")
    logger.info("=" * 60)
    
    available_cameras = []
    max_cameras_to_check = 10  # Check first 10 camera indices
    
    for camera_id in range(max_cameras_to_check):
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            # Try to read a frame to ensure camera works
            ret, frame = cap.read()
            if ret and frame is not None:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                camera_info = {
                    'id': camera_id,
                    'width': width,
                    'height': height,
                    'fps': fps if fps > 0 else 'Unknown'
                }
                available_cameras.append(camera_info)
                
                logger.info(f"Camera {camera_id}: {width}x{height} @ {fps} FPS - AVAILABLE")
            else:
                logger.info(f"Camera {camera_id}: DETECTED but cannot read frames")
            cap.release()
        else:
            logger.debug(f"Camera {camera_id}: NOT AVAILABLE")
    
    if not available_cameras:
        logger.warning("No webcam devices found!")
        logger.info("Make sure your webcam is connected and not used by other applications.")
    else:
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        for cam in available_cameras:
            logger.info(f"Use --camera {cam['id']} for {cam['width']}x{cam['height']} camera")
        logger.info(f"\nExample: python app.py webcam --camera {available_cameras[0]['id']}")
        logger.info("=" * 60)
    
    return available_cameras


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='SVM+ORB Face Detection with Mask Overlay',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model with enhanced pipeline')
    train_parser.add_argument('--pos_dir', type=str, default='data/faces',
                             help='Directory with positive samples (faces)')
    train_parser.add_argument('--neg_dir', type=str, default='data/non_faces',
                             help='Directory with negative samples')
    train_parser.add_argument('--k', type=int, default=256,
                             help='Number of visual words (codebook size) - default k=256 as per prompt')
    train_parser.add_argument('--orb_features', type=int, default=1000,
                             help='Number of ORB features per image (1000 for high quality)')
    train_parser.add_argument('--max_desc', type=int, default=300000,
                             help='Maximum descriptors for K-Means (300k for better quality)')
    train_parser.add_argument('--svm', type=str, default='linear', choices=['linear', 'rbf'],
                             help='SVM kernel type (linear or rbf with 5-fold CV)')
    train_parser.add_argument('--C', type=str, default='0.1,1.0,10.0',
                             help='C values for grid search (comma-separated)')
    train_parser.add_argument('--gamma', type=str, default='scale,auto',
                             help='Gamma values for RBF kernel (comma-separated)')
    train_parser.add_argument('--test_size', type=float, default=0.15,
                             help='Test set proportion (70/15/15 split as per prompt)')
    train_parser.add_argument('--val_size', type=float, default=0.15,
                             help='Validation set proportion (70/15/15 split as per prompt)')
    train_parser.add_argument('--model_dir', type=str, default='models',
                             help='Directory to save models')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
    # Enhanced training arguments - AUGMENTATION DISABLED BY DEFAULT
    train_parser.add_argument('--roi_size', type=int, default=48,
                             help='ROI size for face regions (square) - smaller for speed')
    train_parser.add_argument('--use_landmarks', action='store_true', default=False,
                             help='Use facial landmarks from .pts files (DISABLED by default for simplicity)')
    train_parser.add_argument('--landmarks', dest='use_landmarks', action='store_true',
                             help='Enable facial landmarks usage')
    train_parser.add_argument('--augment_data', action='store_true', default=False,
                             help='Enable data augmentation (DISABLED by default - like successful friend)')
    train_parser.add_argument('--augment', dest='augment_data', action='store_true',
                             help='Enable data augmentation (NOT RECOMMENDED)')
    train_parser.add_argument('--aug_factor', type=int, default=0,
                             help='Number of augmented versions per positive sample (0=disabled, RECOMMENDED)')
    train_parser.add_argument('--handle_imbalance', action='store_true', default=True,
                             help='Handle class imbalance with weighted SVM')
    train_parser.add_argument('--no_imbalance', dest='handle_imbalance', action='store_false',
                             help='Disable class imbalance handling')
    train_parser.add_argument('--max_pos', type=int, default=3000,
                             help='Max positive samples (3000 for higher performance)')
    train_parser.add_argument('--max_neg', type=int, default=800, 
                             help='Max negative samples (800 for 4:1 face ratio)')
    train_parser.add_argument('--fast', action='store_true', default=False,
                             help='Fast training mode (smaller dataset, 3-fold CV)')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--model_dir', type=str, default='models',
                            help='Directory with trained models')
    eval_parser.add_argument('--report', type=str, default='reports/test_metrics.json',
                            help='Path to save evaluation report')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference on image')
    infer_parser.add_argument('--image', type=str, required=True,
                             help='Input image path')
    infer_parser.add_argument('--out', type=str, default='output.jpg',
                             help='Output image path')
    infer_parser.add_argument('--mask', type=str, default=None,
                             help='Path to mask PNG')
    infer_parser.add_argument('--model_dir', type=str, default='models',
                             help='Directory with trained models')
    infer_parser.add_argument('--rotate', action='store_true',
                             help='Enable mask rotation based on eyes')
    infer_parser.add_argument('--boxes', action='store_true', default=True,
                             help='Draw bounding boxes')
    infer_parser.add_argument('--show', action='store_true',
                             help='Show result window')
    
    # Webcam command
    webcam_parser = subparsers.add_parser('webcam', help='Run inference on webcam')
    webcam_parser.add_argument('--camera', type=int, default=1,
                              help='Camera ID (default: 1)')
    webcam_parser.add_argument('--mask_dir', type=str, default='assets',
                              help='Directory containing mask files (mask1.png - mask7.png)')
    webcam_parser.add_argument('--model_dir', type=str, default='models',
                              help='Directory with trained models')
    webcam_parser.add_argument('--rotate', action='store_true',
                              help='Enable mask rotation based on eyes')
    webcam_parser.add_argument('--show', action='store_true', default=True,
                              help='Show live display')
    webcam_parser.add_argument('--out', type=str, default=None,
                              help='Output video path (optional)')
    
    # List webcams command
    list_webcams_parser = subparsers.add_parser('list_webcams', help='List available webcams')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Run command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'eval':
        eval_command(args)
    elif args.command == 'infer':
        infer_command(args)
    elif args.command == 'webcam':
        webcam_command(args)
    elif args.command == 'list_webcams':
        list_webcams_command(args)


if __name__ == '__main__':
    main()

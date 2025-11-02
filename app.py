"""
Main CLI application for SVM+ORB Face Detection with Mask Overlay.

Usage:
    python app.py train --pos_dir data/faces --neg_dir data/non_faces
    python app.py eval --report reports/test_metrics.json
    python app.py infer --image input.jpg --out output.jpg --mask assets/mask.png
    python app.py webcam --camera 0 --mask_dir assets --show

Webcam Controls:
    'q' - Quit
    'm' - Toggle mask on/off
    '1-7' - Switch between mask1.png to mask7.png
    's' - Take screenshot
"""
import argparse
import sys
from pathlib import Path

from pipelines.utils import logger, save_json, set_seed, ensure_dir
from pipelines.dataset import DatasetManager
from pipelines.features import FeaturePipeline
from pipelines.train import train_pipeline, SVMTrainer
from pipelines.infer import ImageInference, VideoInference


def train_command(args):
    """Train the face detection model."""
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Prepare dataset
    logger.info("Step 1: Preparing dataset...")
    dataset_manager = DatasetManager(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    dataset = dataset_manager.prepare_dataset(auto_generate_negatives=True)
    
    # Save dataset splits
    ensure_dir('data')
    dataset_manager.save_dataset(dataset, 'data/dataset_splits.json')
    
    # Get train/val/test splits
    train_rois, train_labels = dataset_manager.get_split_data(dataset, 'train')
    val_rois, val_labels = dataset_manager.get_split_data(dataset, 'val')
    test_rois, test_labels = dataset_manager.get_split_data(dataset, 'test')
    
    logger.info(f"Train: {len(train_rois)}, Val: {len(val_rois)}, Test: {len(test_rois)}")
    
    # Step 2: Feature extraction and BoVW encoding
    logger.info("Step 2: Building feature pipeline...")
    feature_pipeline = FeaturePipeline(
        orb_params={'n_features': args.orb_features},
        bovw_params={'n_clusters': args.k, 'random_state': args.seed}
    )
    
    # Fit on training data
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
    
    # Step 3: Train SVM classifier
    logger.info("Step 3: Training SVM classifier...")
    
    C_values = [float(c) for c in args.C.split(',')]
    gamma_values = None
    if args.svm == 'rbf' and args.gamma:
        gamma_values = args.gamma.split(',')
        # Convert to float if not 'scale' or 'auto'
        gamma_values = [g if g in ['scale', 'auto'] else float(g) for g in gamma_values]
    
    trainer, metrics = train_pipeline(
        X_train, train_labels,
        X_val, val_labels,
        kernel=args.svm,
        C_values=C_values,
        gamma_values=gamma_values,
        model_dir=args.model_dir
    )
    
    # Step 4: Final evaluation on test set
    logger.info("Step 4: Final evaluation on test set...")
    test_metrics = trainer.evaluate(X_test, test_labels, save_dir='reports')
    
    # Save all metrics
    all_metrics = {
        'train': metrics['train'],
        'validation': metrics['validation'],
        'test': test_metrics,
        'config': {
            'pos_dir': args.pos_dir,
            'neg_dir': args.neg_dir,
            'orb_features': args.orb_features,
            'k': args.k,
            'max_desc': args.max_desc,
            'svm_kernel': args.svm,
            'C_values': C_values,
            'gamma_values': gamma_values,
            'seed': args.seed
        }
    }
    
    save_json(all_metrics, 'reports/metrics.json')
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test AUC: {test_metrics['roc_auc']:.4f}")
    logger.info("=" * 60)


def eval_command(args):
    """Evaluate trained model on test set."""
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    # Load dataset
    from pipelines.utils import load_json
    dataset_info = load_json('data/dataset_splits.json')
    
    # Reconstruct dataset
    dataset_manager = DatasetManager(
        pos_dir=dataset_info['metadata']['pos_dir'],
        neg_dir=dataset_info['metadata']['neg_dir'],
        random_state=dataset_info['metadata']['random_state']
    )
    
    dataset = dataset_manager.prepare_dataset()
    test_rois, test_labels = dataset_manager.get_split_data(dataset, 'test')
    
    # Load feature pipeline
    feature_pipeline = FeaturePipeline()
    feature_pipeline.load_codebook(f'{args.model_dir}/codebook.pkl')
    
    # Transform test data
    logger.info("Extracting features from test set...")
    X_test = feature_pipeline.transform(test_rois)
    
    # Load classifier
    classifier = SVMTrainer()
    classifier.load(args.model_dir)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, test_labels, save_dir='reports')
    
    # Save metrics
    if args.report:
        save_json(metrics, args.report)
    
    logger.info("=" * 60)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")
    logger.info(f"AUC: {metrics['roc_auc']:.4f}")
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='SVM+ORB Face Detection with Mask Overlay',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--pos_dir', type=str, default='data/faces',
                             help='Directory with positive samples (faces)')
    train_parser.add_argument('--neg_dir', type=str, default='data/non_faces',
                             help='Directory with negative samples')
    train_parser.add_argument('--k', type=int, default=256,
                             help='Number of visual words (codebook size)')
    train_parser.add_argument('--orb_features', type=int, default=500,
                             help='Number of ORB features per image')
    train_parser.add_argument('--max_desc', type=int, default=200000,
                             help='Maximum descriptors for K-Means')
    train_parser.add_argument('--svm', type=str, default='linear', choices=['linear', 'rbf'],
                             help='SVM kernel type')
    train_parser.add_argument('--C', type=str, default='0.1,1.0,10.0',
                             help='C values for grid search (comma-separated)')
    train_parser.add_argument('--gamma', type=str, default='scale,auto,0.001,0.01',
                             help='Gamma values for RBF kernel (comma-separated)')
    train_parser.add_argument('--test_size', type=float, default=0.15,
                             help='Test set proportion')
    train_parser.add_argument('--val_size', type=float, default=0.15,
                             help='Validation set proportion')
    train_parser.add_argument('--model_dir', type=str, default='models',
                             help='Directory to save models')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
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
    webcam_parser.add_argument('--camera', type=int, default=0,
                              help='Camera ID')
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


if __name__ == '__main__':
    main()

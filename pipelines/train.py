"""
Enhanced SVM training with cross-validation and hyperparameter search.
Includes support for imbalanced datasets and advanced evaluation metrics.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path
from typing import Dict, Tuple
from .utils import logger, Timer, plot_confusion_matrix, plot_pr_curve, plot_roc_curve


class EnhancedSVMTrainer:
    """
    SVM classifier trainer implementing prompt requirements:
    - ORB features → BoVW (k=256) → StandardScaler → SVM
    - 5-fold cross-validation for hyperparameter search  
    - Support for Linear and RBF kernels
    - Handle class imbalance with balanced weights
    - Save models as svm.pkl, scaler.pkl as per prompt
    """
    
    def __init__(self, kernel: str = 'linear', random_state: int = 42, 
                 handle_imbalance: bool = True):
        """
        Initialize SVM trainer following classical computer vision approach.
        
        Args:
            kernel: SVM kernel ('linear' or 'rbf') as per prompt
            random_state: Random seed for reproducibility
            handle_imbalance: Whether to handle class imbalance
        """
        self.kernel = kernel
        self.random_state = random_state
        self.handle_imbalance = handle_imbalance
        self.scaler = StandardScaler()  # For BoVW feature standardization
        self.svm = None
        self.best_params = None
        self.class_weights = None
        logger.info(f"Initialized SVM trainer (classical CV): {kernel} kernel")
        logger.info(f"Handle class imbalance: {handle_imbalance}")
    
    def compute_class_weights(self, y_train: np.ndarray) -> Dict:
        """Compute class weights for imbalanced datasets."""
        if not self.handle_imbalance:
            return None
            
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        
        logger.info(f"Computed class weights: {class_weight_dict}")
        self.class_weights = class_weight_dict
        return class_weight_dict
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              C_values: list = None, gamma_values: list = None,
              cv_folds: int = 5, fast_mode: bool = False) -> Dict:
        """
        Train SVM with cross-validation and hyperparameter search.
        
        Args:
            X_train: Training features (BoVW histograms)
            y_train: Training labels (face=1, non-face=0)
            C_values: List of C values to search (regularization parameter)
            gamma_values: List of gamma values for RBF kernel
            cv_folds: Number of CV folds (fixed at 5 as per prompt requirement)
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training SVM classifier (classical computer vision approach)...")
        logger.info("Pipeline: ORB features → BoVW encoding → StandardScaler → SVM")
        logger.info("SVM Configuration: max_iter=5000, cache_size=1000, tol=1e-4")
        
        # Ensure 5-fold CV as per prompt requirement
        if cv_folds != 5:
            logger.warning(f"CV folds changed from {cv_folds} to 5 (prompt requirement)")
            cv_folds = 5
        
        # Compute class weights for imbalanced datasets
        class_weights = self.compute_class_weights(y_train)
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Class distribution: {class_dist}")
        
        # Scale BoVW features with StandardScaler (as per prompt requirement)
        with Timer("Standardizing BoVW features"):
            X_train_scaled = self.scaler.fit_transform(X_train)
            logger.info(f"Scaled features shape: {X_train_scaled.shape}")
            logger.info(f"Feature mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
        
        # Setup parameter grid for hyperparameter search  
        if C_values is None:
            # Aggressive C search for maximum performance
            if fast_mode:
                C_values = [0.1, 1.0, 10.0]  # Quick search
            else:
                C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]  # Extensive search
        
        param_grid = {'C': C_values}
        
        if self.kernel == 'rbf':
            if gamma_values is None:
                if fast_mode:
                    gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1]
                else:
                    gamma_values = ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
            param_grid['gamma'] = gamma_values
        
        # Keep CV folds consistent (5-fold as per requirements)
        logger.info(f"Using {cv_folds}-fold CV for hyperparameter search")
        
        logger.info(f"SVM kernel: {self.kernel}")
        logger.info(f"Parameter grid: {param_grid} (precision-focused)")
        logger.info(f"Using {cv_folds}-fold CV as per requirements")
        
        # Grid search with specified CV folds (5-fold as per requirements)
        with Timer(f"Hyperparameter search with {cv_folds}-fold CV"):
            svm_base = SVC(
                kernel=self.kernel, 
                random_state=self.random_state, 
                probability=True,
                class_weight=class_weights if self.handle_imbalance else None,
                max_iter=5000,  # Increased for better convergence
                cache_size=1000,  # Increased cache for speed
                tol=1e-4  # Slightly relaxed tolerance for faster convergence
            )
            
            # Use 3-fold CV for speed (compromise)
            cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Sample subset for faster CV if dataset is too large
            if len(X_train_scaled) > 3000:
                logger.info("Large dataset detected - using subset for faster CV")
                from sklearn.model_selection import train_test_split
                X_subset, _, y_subset, _ = train_test_split(
                    X_train_scaled, y_train, 
                    train_size=3000, 
                    stratify=y_train,
                    random_state=self.random_state
                )
                logger.info(f"Using subset: {len(X_subset)} samples for CV")
            else:
                X_subset, y_subset = X_train_scaled, y_train
            
            grid_search = GridSearchCV(
                svm_base,
                param_grid,
                cv=cv_strategy,
                scoring='f1',
                n_jobs=-1,
                verbose=2,  # More verbose for debugging
                return_train_score=True
            )
            
            logger.info("Starting GridSearchCV...")
            grid_search.fit(X_subset, y_subset)
            
            # Retrain on full dataset with best params
            if len(X_train_scaled) > 3000:
                logger.info("Retraining on full dataset with best parameters...")
                best_svm = SVC(**grid_search.best_params_, 
                              kernel=self.kernel,
                              random_state=self.random_state,
                              probability=True,
                              class_weight=class_weights,
                              max_iter=5000,  # Consistent with main training
                              cache_size=1000,
                              tol=1e-4)
                best_svm.fit(X_train_scaled, y_train)
                grid_search.best_estimator_ = best_svm
        
        self.svm = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        
        # Evaluate on training set
        y_pred = self.svm.predict(X_train_scaled)
        y_proba = self.svm.predict_proba(X_train_scaled)[:, 1]
        
        # Comprehensive training metrics
        metrics = {
            'best_params': self.best_params,
            'best_cv_score': float(grid_search.best_score_),
            'train_accuracy': float(accuracy_score(y_train, y_pred)),
            'train_precision': float(precision_score(y_train, y_pred)),
            'train_recall': float(recall_score(y_train, y_pred)),
            'train_f1': float(f1_score(y_train, y_pred)),
            'train_auc': float(roc_auc_score(y_train, y_proba)),
            'train_ap': float(average_precision_score(y_train, y_proba)),
            'class_weights': class_weights,
            'class_distribution': class_dist,
            'cv_scores': {
                'mean': float(grid_search.best_score_),
                'std': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
            }
        }
        
        logger.info("Training Results:")
        logger.info(f"  Accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['train_precision']:.4f}")
        logger.info(f"  Recall: {metrics['train_recall']:.4f}")
        logger.info(f"  F1: {metrics['train_f1']:.4f}")
        logger.info(f"  AUC: {metrics['train_auc']:.4f}")
        logger.info(f"  AP: {metrics['train_ap']:.4f}")
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 save_dir: str = None) -> Dict:
        """
        Evaluate trained SVM on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_dir: Directory to save evaluation plots
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.svm is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Evaluating SVM on test set...")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.svm.predict(X_test_scaled)
        y_proba = self.svm.predict_proba(X_test_scaled)[:, 1]
        
        # Compute metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'average_precision': float(average_precision_score(y_test, y_proba))
        }
        
        logger.info(f"Test metrics: Acc={metrics['accuracy']:.4f}, "
                   f"Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}, "
                   f"F1={metrics['f1']:.4f}, "
                   f"AUC={metrics['roc_auc']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Generate plots if save_dir provided
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            # Confusion matrix
            plot_confusion_matrix(cm, save_path=f"{save_dir}/confusion_matrix.png")
            
            # PR curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            plot_pr_curve(precision, recall, metrics['average_precision'],
                         save_path=f"{save_dir}/pr_curve.png")
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plot_roc_curve(fpr, tpr, metrics['roc_auc'],
                          save_path=f"{save_dir}/roc_curve.png")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.svm is None:
            raise ValueError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.svm is None:
            raise ValueError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        return self.svm.predict_proba(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values."""
        if self.svm is None:
            raise ValueError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        return self.svm.decision_function(X_scaled)
    
    def save(self, model_dir: str):
        """
        Save trained models following prompt requirements.
        Saves: svm.pkl, scaler.pkl (codebook.pkl saved separately by FeaturePipeline)
        
        Args:
            model_dir: Directory to save models
        """
        if self.svm is None:
            raise ValueError("No model to save")
        
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save SVM classifier (as per prompt: svm.pkl)
        svm_path = f"{model_dir}/svm.pkl"
        joblib.dump(self.svm, svm_path)
        logger.info(f"Saved SVM classifier to {svm_path}")
        
        # Save StandardScaler (as per prompt: scaler.pkl)  
        scaler_path = f"{model_dir}/scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved StandardScaler to {scaler_path}")
        
        logger.info("Model files saved (classical CV pipeline):")
        logger.info(f"  - {svm_path} (SVM classifier)")
        logger.info(f"  - {scaler_path} (BoVW feature scaler)")
        logger.info(f"  - codebook.pkl (saved separately by FeaturePipeline)")
    
    def load(self, model_dir: str):
        """
        Load trained model and scaler.
        
        Args:
            model_dir: Directory containing models
        """
        svm_path = f"{model_dir}/svm.pkl"
        scaler_path = f"{model_dir}/scaler.pkl"
        
        self.svm = joblib.load(svm_path)
        self.scaler = joblib.load(scaler_path)
        
        logger.info(f"Loaded SVM from {svm_path}")
        logger.info(f"Loaded scaler from {scaler_path}")


def enhanced_train_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           kernel: str = 'linear',
                           C_values: list = None,
                           gamma_values: list = None,
                           model_dir: str = 'models',
                           handle_imbalance: bool = True,
                           fast_mode: bool = False) -> Tuple[EnhancedSVMTrainer, Dict]:
    """
    Complete SVM training pipeline following prompt requirements:
    - ORB features → BoVW encoding → StandardScaler → SVM classifier
    - Cross-validation for hyperparameter search (7-fold for quality, 3-fold for speed)
    - Support for both Linear and RBF kernels
    - Handle class imbalance with balanced weights
    
    Args:
        X_train: Training features (BoVW histograms)
        y_train: Training labels (face=1, non-face=0)
        X_val: Validation features (BoVW histograms)
        y_val: Validation labels
        kernel: SVM kernel ('linear' or 'rbf')
        C_values: C values for hyperparameter search
        gamma_values: Gamma values for RBF kernel search
        model_dir: Directory to save trained models
        handle_imbalance: Whether to use balanced class weights
        fast_mode: Whether to use fast training (3-fold CV, smaller param grid)
    
    Returns:
        (trained EnhancedSVMTrainer, combined train/validation metrics)
    """
    logger.info("=" * 60)
    logger.info("SVM TRAINING PIPELINE (Classical Computer Vision)")
    logger.info("=" * 60)
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Feature dimensions: {X_train.shape[1]} (BoVW histogram)")
    logger.info(f"SVM kernel: {kernel}")
    
    # Train with enhanced CV (5-fold as per requirements)
    trainer = EnhancedSVMTrainer(kernel=kernel, handle_imbalance=handle_imbalance)
    train_metrics = trainer.train(X_train, y_train, C_values, gamma_values, 
                                cv_folds=5, fast_mode=fast_mode)
    
    # Validate on held-out validation set
    logger.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate(X_val, y_val)
    
    # Save models (svm.pkl, scaler.pkl as per prompt)
    trainer.save(model_dir)
    
    # Combine metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics
    }
    
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info(f"Best CV F1: {train_metrics['best_cv_score']:.4f}")
    logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
    logger.info(f"Validation AUC: {val_metrics['roc_auc']:.4f}")
    logger.info("=" * 60)
    
    return trainer, all_metrics


# Backward compatibility
SVMTrainer = EnhancedSVMTrainer


def train_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   kernel: str = 'linear',
                   C_values: list = None,
                   gamma_values: list = None,
                   model_dir: str = 'models') -> Tuple[EnhancedSVMTrainer, Dict]:
    """
    Complete training pipeline with validation (backward compatibility).
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        kernel: SVM kernel
        C_values: C values for grid search
        gamma_values: Gamma values for grid search
        model_dir: Directory to save models
    
    Returns:
        (trained EnhancedSVMTrainer, validation metrics)
    """
    return enhanced_train_pipeline(
        X_train, y_train, X_val, y_val,
        kernel, C_values, gamma_values, model_dir,
        handle_imbalance=True
    )

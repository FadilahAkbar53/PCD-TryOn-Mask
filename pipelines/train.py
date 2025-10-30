"""
SVM training with cross-validation and hyperparameter search.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
import joblib
from pathlib import Path
from typing import Dict, Tuple
from .utils import logger, Timer, plot_confusion_matrix, plot_pr_curve, plot_roc_curve


class SVMTrainer:
    """SVM classifier trainer with hyperparameter optimization."""
    
    def __init__(self, kernel: str = 'linear', random_state: int = 42):
        """
        Initialize SVM trainer.
        
        Args:
            kernel: SVM kernel ('linear' or 'rbf')
            random_state: Random seed
        """
        self.kernel = kernel
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.svm = None
        self.best_params = None
        logger.info(f"Initialized SVM trainer with {kernel} kernel")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              C_values: list = None, gamma_values: list = None,
              cv_folds: int = 5) -> Dict:
        """
        Train SVM with hyperparameter search using cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            C_values: List of C values to search
            gamma_values: List of gamma values (for RBF kernel)
            cv_folds: Number of CV folds
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training SVM classifier...")
        
        # Scale features
        with Timer("Feature scaling"):
            X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Setup parameter grid
        if C_values is None:
            C_values = [0.1, 1.0, 10.0]
        
        param_grid = {'C': C_values}
        
        if self.kernel == 'rbf':
            if gamma_values is None:
                gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1]
            param_grid['gamma'] = gamma_values
        
        logger.info(f"Parameter grid: {param_grid}")
        
        # Grid search with cross-validation
        with Timer("Hyperparameter search"):
            svm_base = SVC(kernel=self.kernel, random_state=self.random_state, probability=True)
            
            grid_search = GridSearchCV(
                svm_base,
                param_grid,
                cv=cv_folds,
                scoring='f1',
                n_jobs=-1,
                verbose=2
            )
            
            grid_search.fit(X_train_scaled, y_train)
        
        self.svm = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        
        # Evaluate on training set
        y_pred = self.svm.predict(X_train_scaled)
        y_proba = self.svm.predict_proba(X_train_scaled)[:, 1]
        
        metrics = {
            'best_params': self.best_params,
            'best_cv_score': float(grid_search.best_score_),
            'train_accuracy': float(accuracy_score(y_train, y_pred)),
            'train_precision': float(precision_score(y_train, y_pred)),
            'train_recall': float(recall_score(y_train, y_pred)),
            'train_f1': float(f1_score(y_train, y_pred)),
            'train_auc': float(roc_auc_score(y_train, y_proba))
        }
        
        logger.info(f"Training metrics: Acc={metrics['train_accuracy']:.4f}, "
                   f"F1={metrics['train_f1']:.4f}, AUC={metrics['train_auc']:.4f}")
        
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
        Save trained model and scaler.
        
        Args:
            model_dir: Directory to save models
        """
        if self.svm is None:
            raise ValueError("No model to save")
        
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save SVM
        svm_path = f"{model_dir}/svm.pkl"
        joblib.dump(self.svm, svm_path)
        logger.info(f"Saved SVM to {svm_path}")
        
        # Save scaler
        scaler_path = f"{model_dir}/scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
    
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


def train_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   kernel: str = 'linear',
                   C_values: list = None,
                   gamma_values: list = None,
                   model_dir: str = 'models') -> Tuple[SVMTrainer, Dict]:
    """
    Complete training pipeline with validation.
    
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
        (trained SVMTrainer, validation metrics)
    """
    # Train
    trainer = SVMTrainer(kernel=kernel)
    train_metrics = trainer.train(X_train, y_train, C_values, gamma_values)
    
    # Validate
    logger.info("Validating on validation set...")
    val_metrics = trainer.evaluate(X_val, y_val)
    
    # Save
    trainer.save(model_dir)
    
    # Combine metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics
    }
    
    return trainer, all_metrics

"""
Sentinel Vision — Face Classifier (SVM / KNN)
===============================================
Trains and runs a classifier on top of FaceNet embeddings.

Key ML concepts demonstrated:
- SVM with RBF kernel for classification in embedding space
- KNN as a non-parametric baseline
- Hyperparameter tuning via GridSearchCV
- Cross-validation for model selection
- Probability calibration for confidence scores
"""

import numpy as np
import yaml
import os
import joblib
from typing import Optional, Tuple, List, Dict
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FaceClassifier:
    """
    Classifies face embeddings into known identities.
    
    This is a TRAINED ML model (the key differentiator for the resume).
    Instead of simple distance thresholding (what the old code did),
    we train an SVM/KNN that learns the decision boundary between
    identities in the 512-d embedding space.
    
    Advantages over raw distance matching:
    - Handles multi-class classification naturally
    - SVM finds optimal separating hyperplanes
    - Provides calibrated probability scores
    - Can be cross-validated and properly evaluated
    
    Args:
        config_path: Path to config.yaml
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.rec_config = config['recognition']
        self.train_config = config['training']['face_classifier']
        
        self.classifier_type = self.rec_config['classifier_type']
        self.classifier_path = self.rec_config['classifier_path']
        self.unknown_threshold = self.rec_config['unknown_threshold']
        self.knn_neighbors = self.rec_config['knn_neighbors']
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.classes_ = []
        
        # Training metrics (stored for reporting)
        self.training_metrics = {}
    
    def _create_svm(self) -> Pipeline:
        """Create SVM pipeline with scaling and probability calibration."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                probability=True,    # Enable probability estimates
                class_weight='balanced',  # Handle class imbalance
                decision_function_shape='ovr'
            ))
        ])
    
    def _create_knn(self) -> Pipeline:
        """Create KNN pipeline with scaling."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(
                n_neighbors=self.knn_neighbors,
                weights='distance',  # Weight by inverse distance
                metric='euclidean'
            ))
        ])
    
    def train(self, embeddings: np.ndarray, labels: np.ndarray,
              do_grid_search: bool = True) -> Dict:
        """
        Train the face classifier.
        
        Args:
            embeddings: (N, 512) array of face embeddings
            labels: (N,) array of string labels (person names)
            do_grid_search: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary of training metrics
        """
        print(f"\n{'='*60}")
        print(f"Training Face Classifier ({self.classifier_type.upper()})")
        print(f"{'='*60}")
        print(f"  Samples:  {len(embeddings)}")
        print(f"  Classes:  {len(np.unique(labels))}")
        print(f"  Features: {embeddings.shape[1]}")
        
        # Encode string labels to integers
        y_encoded = self.label_encoder.fit_transform(labels)
        self.classes_ = list(self.label_encoder.classes_)
        
        print(f"  Classes:  {self.classes_}")
        
        if self.classifier_type == 'svm':
            self.model = self._create_svm()
            
            if do_grid_search:
                print("\n  Running GridSearchCV for SVM...")
                param_grid = {
                    'svm__C': [0.1, 1, 10, 100],
                    'svm__gamma': ['scale', 'auto', 0.001, 0.01],
                    'svm__kernel': ['rbf', 'linear']
                }
                
                grid_search = GridSearchCV(
                    self.model, param_grid,
                    cv=min(self.train_config['cross_validation_folds'], len(np.unique(y_encoded))),
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(embeddings, y_encoded)
                
                self.model = grid_search.best_estimator_
                self.training_metrics['best_params'] = grid_search.best_params_
                self.training_metrics['best_cv_score'] = grid_search.best_score_
                
                print(f"  Best params: {grid_search.best_params_}")
                print(f"  Best CV score: {grid_search.best_score_:.4f}")
            else:
                self.model.fit(embeddings, y_encoded)
        
        elif self.classifier_type == 'knn':
            self.model = self._create_knn()
            
            if do_grid_search:
                print("\n  Running GridSearchCV for KNN...")
                param_grid = {
                    'knn__n_neighbors': [3, 5, 7, 9, 11],
                    'knn__weights': ['uniform', 'distance'],
                    'knn__metric': ['euclidean', 'cosine']
                }
                
                grid_search = GridSearchCV(
                    self.model, param_grid,
                    cv=min(self.train_config['cross_validation_folds'], len(np.unique(y_encoded))),
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(embeddings, y_encoded)
                
                self.model = grid_search.best_estimator_
                self.training_metrics['best_params'] = grid_search.best_params_
                self.training_metrics['best_cv_score'] = grid_search.best_score_
                
                print(f"  Best params: {grid_search.best_params_}")
                print(f"  Best CV score: {grid_search.best_score_:.4f}")
            else:
                self.model.fit(embeddings, y_encoded)
        
        # Cross-validation scores
        cv_folds = min(self.train_config['cross_validation_folds'], len(np.unique(y_encoded)))
        if cv_folds >= 2:
            cv_scores = cross_val_score(self.model, embeddings, y_encoded, cv=cv_folds)
            self.training_metrics['cv_scores'] = cv_scores.tolist()
            self.training_metrics['cv_mean'] = float(cv_scores.mean())
            self.training_metrics['cv_std'] = float(cv_scores.std())
            print(f"\n  Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.is_trained = True
        self.training_metrics['n_samples'] = len(embeddings)
        self.training_metrics['n_classes'] = len(self.classes_)
        self.training_metrics['classifier_type'] = self.classifier_type
        
        return self.training_metrics
    
    def predict(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Predict identity from a face embedding.
        
        Args:
            embedding: (512,) face embedding vector
            
        Returns:
            (predicted_name, confidence) tuple.
            Returns ("Unknown", 0.0) if confidence is below threshold.
        """
        if not self.is_trained:
            return "Unknown", 0.0
        
        # Reshape for single prediction
        X = embedding.reshape(1, -1)
        
        # Get probability scores for all classes
        probabilities = self.model.predict_proba(X)[0]
        
        # Get best prediction
        best_idx = np.argmax(probabilities)
        best_prob = probabilities[best_idx]
        
        # Apply confidence threshold
        if best_prob < self.unknown_threshold:
            return "Unknown", float(best_prob)
        
        predicted_label = self.label_encoder.inverse_transform([best_idx])[0]
        return str(predicted_label), float(best_prob)
    
    def predict_batch(self, embeddings: np.ndarray) -> List[Tuple[str, float]]:
        """
        Predict identities for a batch of embeddings.
        
        Args:
            embeddings: (N, 512) array of face embeddings
            
        Returns:
            List of (predicted_name, confidence) tuples
        """
        if not self.is_trained:
            return [("Unknown", 0.0)] * len(embeddings)
        
        probabilities = self.model.predict_proba(embeddings)
        
        results = []
        for probs in probabilities:
            best_idx = np.argmax(probs)
            best_prob = probs[best_idx]
            
            if best_prob < self.unknown_threshold:
                results.append(("Unknown", float(best_prob)))
            else:
                name = self.label_encoder.inverse_transform([best_idx])[0]
                results.append((str(name), float(best_prob)))
        
        return results
    
    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Evaluate classifier on a test set.
        
        Args:
            embeddings: (N, 512) test embeddings
            labels: (N,) true labels
            
        Returns:
            Dictionary with classification report and confusion matrix
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        y_true = self.label_encoder.transform(labels)
        y_pred = self.model.predict(embeddings)
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.classes_,
            output_dict=True
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Print report
        print(f"\n{'='*60}")
        print("Classification Report")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred, target_names=self.classes_))
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': report['accuracy'],
            'classes': self.classes_
        }
    
    def save(self, path: Optional[str] = None):
        """Save trained model to disk."""
        save_path = path or self.classifier_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'classes': self.classes_,
            'unknown_threshold': self.unknown_threshold,
            'training_metrics': self.training_metrics,
            'classifier_type': self.classifier_type
        }
        
        joblib.dump(model_data, save_path)
        print(f"[Saved] Face classifier to: {save_path}")
    
    def load(self, path: Optional[str] = None):
        """Load a trained model from disk."""
        load_path = path or self.classifier_path
        
        if not os.path.exists(load_path):
            print(f"[Warning] No model found at {load_path}")
            return False
        
        model_data = joblib.load(load_path)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.classes_ = model_data['classes']
        self.unknown_threshold = model_data.get('unknown_threshold', 0.5)
        self.training_metrics = model_data.get('training_metrics', {})
        self.classifier_type = model_data.get('classifier_type', 'svm')
        self.is_trained = True
        
        print(f"[Loaded] Face classifier from: {load_path}")
        print(f"  Type: {self.classifier_type}")
        print(f"  Classes: {self.classes_}")
        
        return True

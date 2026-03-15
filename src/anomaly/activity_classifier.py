"""
Sentinel Vision — Activity Classifier
=======================================
Classifies human activities from pose features.

Key ML concepts demonstrated:
- Random Forest ensemble with feature importance analysis
- LSTM for temporal sequence classification
- Multi-class classification
- Feature importance visualization
"""

import numpy as np
import yaml
import os
import joblib
from typing import Optional, Tuple, List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


class ActivityClassifier:
    """
    Classifies human activities based on pose features.
    
    Categories: normal_walking, running, loitering, suspicious, falling
    
    Supports two model types:
    - Random Forest: Works on single-frame pose features
    - LSTM: Works on sequences of pose features (temporal model)
    
    Args:
        config_path: Path to config.yaml
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.act_config = config['activity']
        self.train_config = config['training']['activity_classifier']
        
        self.classifier_type = self.act_config['classifier_type']
        self.classifier_path = self.act_config['classifier_path']
        self.categories = self.act_config['categories']
        self.sequence_length = self.act_config['sequence_length']
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importances_ = None
        self.training_metrics = {}
    
    def _create_random_forest(self) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        return RandomForestClassifier(
            n_estimators=self.train_config['n_estimators'],
            max_depth=self.train_config['max_depth'],
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, features: np.ndarray, labels: np.ndarray,
              do_grid_search: bool = True) -> Dict:
        """
        Train the activity classifier.
        
        Args:
            features: (N, F) feature matrix from PoseEstimator
            labels: (N,) activity category labels
            do_grid_search: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary of training metrics
        """
        print(f"\n{'='*60}")
        print(f"Training Activity Classifier ({self.classifier_type.upper()})")
        print(f"{'='*60}")
        print(f"  Samples:    {len(features)}")
        print(f"  Features:   {features.shape[1]}")
        print(f"  Categories: {list(np.unique(labels))}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        if self.classifier_type == 'random_forest':
            if do_grid_search:
                print("\n  Running GridSearchCV for Random Forest...")
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                base_model = self._create_random_forest()
                n_folds = min(5, len(np.unique(y_encoded)))
                
                grid_search = GridSearchCV(
                    base_model, param_grid,
                    cv=n_folds,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_scaled, y_encoded)
                
                self.model = grid_search.best_estimator_
                self.training_metrics['best_params'] = grid_search.best_params_
                self.training_metrics['best_cv_score'] = float(grid_search.best_score_)
                
                print(f"  Best params: {grid_search.best_params_}")
                print(f"  Best CV F1: {grid_search.best_score_:.4f}")
            else:
                self.model = self._create_random_forest()
                self.model.fit(X_scaled, y_encoded)
            
            # Feature importance
            self.feature_importances_ = self.model.feature_importances_
        
        elif self.classifier_type == 'lstm':
            self._train_lstm(X_scaled, y_encoded)
        
        # Cross-validation
        n_folds = min(5, len(np.unique(y_encoded)))
        if n_folds >= 2 and self.classifier_type == 'random_forest':
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=n_folds, scoring='f1_weighted')
            self.training_metrics['cv_scores'] = cv_scores.tolist()
            self.training_metrics['cv_mean'] = float(cv_scores.mean())
            self.training_metrics['cv_std'] = float(cv_scores.std())
            print(f"\n  Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.is_trained = True
        self.training_metrics['n_samples'] = len(features)
        self.training_metrics['n_features'] = features.shape[1]
        self.training_metrics['n_classes'] = len(np.unique(labels))
        
        return self.training_metrics
    
    def _train_lstm(self, X: np.ndarray, y: np.ndarray):
        """
        Train an LSTM model on pose feature sequences.
        
        This provides a temporal model baseline to compare against
        the single-frame Random Forest approach.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            print("[Warning] PyTorch not available. Falling back to Random Forest.")
            self.classifier_type = 'random_forest'
            self.model = self._create_random_forest()
            self.model.fit(X, y)
            return
        
        # Reshape for LSTM: (N, seq_len, features)
        n_features = X.shape[1]
        seq_len = self.sequence_length
        
        # Create sequences from feature vectors
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_len + 1):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len - 1])
        
        if len(X_seq) == 0:
            print("[Warning] Not enough data for LSTM sequences. Falling back to RF.")
            self.classifier_type = 'random_forest'
            self.model = self._create_random_forest()
            self.model.fit(X, y)
            return
        
        X_tensor = torch.FloatTensor(np.array(X_seq))
        y_tensor = torch.LongTensor(np.array(y_seq))
        
        n_classes = len(np.unique(y))
        hidden_size = self.train_config['lstm_hidden_size']
        
        # Define LSTM model
        class ActivityLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                                   batch_first=True, dropout=0.3)
                self.fc1 = nn.Linear(hidden_size, 64)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(64, num_classes)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_output = lstm_out[:, -1, :]
                x = self.dropout(self.relu(self.fc1(last_output)))
                return self.fc2(x)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lstm_model = ActivityLSTM(n_features, hidden_size, n_classes).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.train_config['lstm_batch_size'], shuffle=True)
        
        # Training loop
        epochs = self.train_config['lstm_epochs']
        print(f"\n  Training LSTM ({epochs} epochs)...")
        
        for epoch in range(epochs):
            lstm_model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            if (epoch + 1) % 10 == 0:
                acc = 100.0 * correct / total
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/len(loader):.4f}, Acc={acc:.1f}%")
        
        self.model = lstm_model
        self.training_metrics['lstm_final_loss'] = total_loss / len(loader)
        self.training_metrics['lstm_final_acc'] = 100.0 * correct / total
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict activity from pose features.
        
        Args:
            features: Single feature vector from PoseEstimator
            
        Returns:
            (activity_label, confidence)
        """
        if not self.is_trained:
            return "unknown", 0.0
        
        X = self.scaler.transform(features.reshape(1, -1))
        
        if self.classifier_type == 'random_forest':
            probs = self.model.predict_proba(X)[0]
            best_idx = np.argmax(probs)
            confidence = float(probs[best_idx])
            label = self.label_encoder.inverse_transform([best_idx])[0]
            return str(label), confidence
        
        elif self.classifier_type == 'lstm':
            # For LSTM, would need sequence buffer (simplified here)
            import torch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).unsqueeze(0)  # Add seq dim
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                best_idx = torch.argmax(probs).item()
                confidence = float(probs[best_idx])
                label = self.label_encoder.inverse_transform([best_idx])[0]
                return str(label), confidence
        
        return "unknown", 0.0
    
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate on test set."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        
        X = self.scaler.transform(features)
        y_true = self.label_encoder.transform(labels)
        y_pred = self.model.predict(X)
        
        classes = list(self.label_encoder.classes_)
        
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n{'='*60}")
        print("Activity Classification Report")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred, target_names=classes))
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': report['accuracy'],
            'classes': classes,
            'feature_importances': self.feature_importances_.tolist() if self.feature_importances_ is not None else None
        }
    
    def save(self, path: Optional[str] = None):
        """Save trained model."""
        save_path = path or self.classifier_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'classifier_type': self.classifier_type,
            'categories': self.categories,
            'feature_importances': self.feature_importances_,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, save_path)
        print(f"[Saved] Activity classifier to: {save_path}")
    
    def load(self, path: Optional[str] = None):
        """Load a trained model."""
        load_path = path or self.classifier_path
        
        if not os.path.exists(load_path):
            print(f"[Warning] No model found at {load_path}")
            return False
        
        model_data = joblib.load(load_path)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.classifier_type = model_data.get('classifier_type', 'random_forest')
        self.feature_importances_ = model_data.get('feature_importances', None)
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = True
        
        print(f"[Loaded] Activity classifier from: {load_path}")
        return True

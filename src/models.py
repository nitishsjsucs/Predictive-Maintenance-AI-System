"""
Machine Learning Models for Predictive Maintenance
===================================================
Advanced ML models with hyperparameter optimization and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import joblib
import json

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier,
    ExtraTreesClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """Container for model evaluation results."""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str
    feature_importance: Optional[Dict] = None
    training_time: float = 0.0


class PredictiveMaintenanceModels:
    """
    Collection of ML models for predictive maintenance.
    Includes automated model selection and hyperparameter tuning.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def get_base_models(self) -> Dict[str, Any]:
        """Get dictionary of base models for comparison."""
        return {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state, n_estimators=100, class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state, n_estimators=100
            ),
            'Extra Trees': ExtraTreesClassifier(
                random_state=self.random_state, n_estimators=100, class_weight='balanced'
            ),
            'AdaBoost': AdaBoostClassifier(
                random_state=self.random_state, n_estimators=100
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state, class_weight='balanced'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(
                random_state=self.random_state, probability=True, class_weight='balanced'
            ),
            'Neural Network': MLPClassifier(
                random_state=self.random_state, hidden_layer_sizes=(100, 50),
                max_iter=500, early_stopping=True
            )
        }
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                         method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using various techniques.
        
        Args:
            X: Features
            y: Target
            method: 'smote', 'adasyn', 'borderline', 'smotetomek', 'smoteenn'
        """
        samplers = {
            'smote': SMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'borderline': BorderlineSMOTE(random_state=self.random_state),
            'smotetomek': SMOTETomek(random_state=self.random_state),
            'smoteenn': SMOTEENN(random_state=self.random_state)
        }
        
        sampler = samplers.get(method, samplers['smote'])
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        print(f"✓ Resampling with {method}:")
        print(f"  Before: {dict(pd.Series(y).value_counts())}")
        print(f"  After: {dict(pd.Series(y_resampled).value_counts())}")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           models: Optional[Dict] = None) -> Dict[str, ModelResult]:
        """
        Train and evaluate multiple models.
        """
        if models is None:
            models = self.get_base_models()
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            import time
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            result = ModelResult(
                name=name,
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                f1=f1_score(y_test, y_pred, zero_division=0),
                roc_auc=roc_auc_score(y_test, y_prob),
                confusion_matrix=confusion_matrix(y_test, y_pred),
                classification_report=classification_report(y_test, y_pred),
                training_time=training_time
            )
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                result.feature_importance = dict(zip(
                    X_train.columns, model.feature_importances_
                ))
            elif hasattr(model, 'coef_'):
                result.feature_importance = dict(zip(
                    X_train.columns, np.abs(model.coef_[0])
                ))
            
            results[name] = result
            self.models[name] = model
            
            print(f"  Accuracy: {result.accuracy:.4f}")
            print(f"  F1-Score: {result.f1:.4f}")
            print(f"  ROC-AUC: {result.roc_auc:.4f}")
        
        self.results = results
        return results
    
    def select_best_model(self, metric: str = 'f1') -> Tuple[str, Any]:
        """Select the best model based on specified metric."""
        if not self.results:
            raise ValueError("No models trained yet. Run train_and_evaluate first.")
        
        best_name = max(self.results.keys(), 
                       key=lambda x: getattr(self.results[x], metric))
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\n★ Best Model: {best_name}")
        print(f"  {metric}: {getattr(self.results[best_name], metric):.4f}")
        
        return best_name, self.best_model
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                              model_name: str = 'Random Forest',
                              search_type: str = 'random',
                              n_iter: int = 50) -> Any:
        """
        Perform hyperparameter tuning on specified model.
        """
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Extra Trees': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        }
        
        base_models = self.get_base_models()
        if model_name not in base_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = base_models[model_name]
        param_grid = param_grids.get(model_name, {})
        
        if not param_grid:
            print(f"No parameter grid defined for {model_name}")
            return model
        
        print(f"\nHyperparameter tuning for {model_name}...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        if search_type == 'random':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv,
                scoring='f1', random_state=self.random_state, n_jobs=-1
            )
        else:
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring='f1', n_jobs=-1
            )
        
        search.fit(X_train, y_train)
        
        print(f"  Best parameters: {search.best_params_}")
        print(f"  Best CV F1-Score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def create_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                        method: str = 'voting') -> Any:
        """
        Create ensemble model from top performers.
        """
        # Get top 3 models
        top_models = sorted(
            self.results.items(), 
            key=lambda x: x[1].f1, 
            reverse=True
        )[:3]
        
        estimators = [(name, self.models[name]) for name, _ in top_models]
        
        print(f"\nCreating {method} ensemble with: {[e[0] for e in estimators]}")
        
        if method == 'voting':
            ensemble = VotingClassifier(
                estimators=estimators, voting='soft'
            )
        elif method == 'stacking':
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series,
                       cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation and return metrics."""
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, 
                                      random_state=self.random_state)
        
        scores = {
            'accuracy': cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy'),
            'f1': cross_val_score(model, X, y, cv=cv_strategy, scoring='f1'),
            'roc_auc': cross_val_score(model, X, y, cv=cv_strategy, scoring='roc_auc'),
            'precision': cross_val_score(model, X, y, cv=cv_strategy, scoring='precision'),
            'recall': cross_val_score(model, X, y, cv=cv_strategy, scoring='recall')
        }
        
        results = {}
        for metric, values in scores.items():
            results[f'{metric}_mean'] = values.mean()
            results[f'{metric}_std'] = values.std()
        
        return results
    
    def save_model(self, model: Any, filepath: str):
        """Save trained model to file."""
        joblib.dump(model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """Load trained model from file."""
        model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of all model results as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for name, result in self.results.items():
            data.append({
                'Model': name,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1-Score': result.f1,
                'ROC-AUC': result.roc_auc,
                'Training Time (s)': result.training_time
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('F1-Score', ascending=False).reset_index(drop=True)


class FailureTypeClassifier:
    """
    Multi-label classifier for predicting specific failure types.
    Predicts: TWF, HDF, PWF, OSF, RNF
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
    def train(self, X: pd.DataFrame, y_multi: pd.DataFrame):
        """
        Train separate classifier for each failure type.
        
        Args:
            X: Features
            y_multi: DataFrame with columns for each failure type
        """
        for failure_type in self.failure_types:
            if failure_type not in y_multi.columns:
                continue
                
            print(f"\nTraining classifier for {failure_type}...")
            
            y = y_multi[failure_type]
            
            # Handle imbalance
            smote = SMOTE(random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X, y)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                class_weight='balanced'
            )
            model.fit(X_res, y_res)
            
            self.models[failure_type] = model
            
            # Evaluate
            y_pred = model.predict(X)
            print(f"  F1-Score: {f1_score(y, y_pred):.4f}")
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict all failure types."""
        predictions = pd.DataFrame(index=range(len(X)))
        
        for failure_type, model in self.models.items():
            predictions[failure_type] = model.predict(X)
            predictions[f'{failure_type}_prob'] = model.predict_proba(X)[:, 1]
        
        return predictions
    
    def get_failure_risk_score(self, X: pd.DataFrame) -> pd.Series:
        """Calculate overall failure risk score."""
        predictions = self.predict(X)
        prob_cols = [col for col in predictions.columns if col.endswith('_prob')]
        return predictions[prob_cols].max(axis=1)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, weights=[0.9, 0.1],
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train models
    pm_models = PredictiveMaintenanceModels()
    
    # Handle imbalance
    X_train_res, y_train_res = pm_models.handle_imbalance(X_train, y_train)
    
    # Train and evaluate
    results = pm_models.train_and_evaluate(X_train_res, X_test, y_train_res, y_test)
    
    # Get best model
    best_name, best_model = pm_models.select_best_model(metric='f1')
    
    # Print summary
    print("\n" + "="*50)
    print(pm_models.get_results_summary().to_string())

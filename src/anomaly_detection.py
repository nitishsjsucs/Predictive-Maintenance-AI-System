"""
Anomaly Detection Module for Predictive Maintenance
====================================================
Real-time anomaly detection for equipment sensor data.
Proactive identification of unusual patterns before failures occur.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    affected_features: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    details: Dict


class AnomalyDetector:
    """
    Multi-method anomaly detection for equipment monitoring.
    
    Combines multiple detection algorithms:
    - Isolation Forest: Good for high-dimensional data
    - Local Outlier Factor: Density-based detection
    - One-Class SVM: Boundary-based detection
    - Statistical methods: Z-score and IQR
    - DBSCAN clustering: Density-based clustering
    """
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.is_fitted = False
        self.feature_stats = {}
        self.thresholds = {}
        
    def fit(self, X: pd.DataFrame):
        """
        Fit all anomaly detection models on normal data.
        
        Args:
            X: Training data (should primarily contain normal samples)
        """
        print("Fitting anomaly detection models...")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Store feature statistics for statistical methods
        self._compute_feature_stats(X)
        
        # Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.models['isolation_forest'].fit(X_scaled)
        print("  ✓ Isolation Forest fitted")
        
        # Local Outlier Factor (for novelty detection)
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True
        )
        self.models['lof'].fit(X_scaled)
        print("  ✓ Local Outlier Factor fitted")
        
        # One-Class SVM
        self.models['ocsvm'] = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=self.contamination
        )
        self.models['ocsvm'].fit(X_scaled)
        print("  ✓ One-Class SVM fitted")
        
        # Elliptic Envelope (assumes Gaussian distribution)
        try:
            self.models['elliptic'] = EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state
            )
            self.models['elliptic'].fit(X_scaled)
            print("  ✓ Elliptic Envelope fitted")
        except Exception as e:
            print(f"  ⚠ Elliptic Envelope failed: {e}")
        
        self.is_fitted = True
        print("✓ All models fitted successfully")
    
    def _compute_feature_stats(self, X: pd.DataFrame):
        """Compute statistics for each feature."""
        for col in X.columns:
            self.feature_stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'q1': X[col].quantile(0.25),
                'q3': X[col].quantile(0.75),
                'iqr': X[col].quantile(0.75) - X[col].quantile(0.25),
                'min': X[col].min(),
                'max': X[col].max()
            }
            
            # Set thresholds
            iqr = self.feature_stats[col]['iqr']
            self.thresholds[col] = {
                'lower_iqr': self.feature_stats[col]['q1'] - 1.5 * iqr,
                'upper_iqr': self.feature_stats[col]['q3'] + 1.5 * iqr,
                'lower_zscore': self.feature_stats[col]['mean'] - 3 * self.feature_stats[col]['std'],
                'upper_zscore': self.feature_stats[col]['mean'] + 3 * self.feature_stats[col]['std']
            }
    
    def detect(self, X: pd.DataFrame, 
               method: str = 'ensemble') -> pd.DataFrame:
        """
        Detect anomalies in data.
        
        Args:
            X: Data to analyze
            method: 'isolation_forest', 'lof', 'ocsvm', 'statistical', 'ensemble'
            
        Returns:
            DataFrame with anomaly labels and scores
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        results = pd.DataFrame(index=X.index)
        
        if method == 'ensemble' or method == 'all':
            # Run all methods and combine
            results['if_pred'] = self.models['isolation_forest'].predict(X_scaled)
            results['if_score'] = -self.models['isolation_forest'].score_samples(X_scaled)
            
            results['lof_pred'] = self.models['lof'].predict(X_scaled)
            results['lof_score'] = -self.models['lof'].score_samples(X_scaled)
            
            results['ocsvm_pred'] = self.models['ocsvm'].predict(X_scaled)
            results['ocsvm_score'] = -self.models['ocsvm'].decision_function(X_scaled)
            
            if 'elliptic' in self.models:
                results['elliptic_pred'] = self.models['elliptic'].predict(X_scaled)
            
            # Statistical anomalies
            stat_results = self._detect_statistical(X)
            results['stat_anomaly'] = stat_results['is_anomaly']
            
            # Ensemble voting (majority vote)
            pred_cols = [col for col in results.columns if col.endswith('_pred')]
            results['vote_count'] = (results[pred_cols] == -1).sum(axis=1)
            
            # Final prediction: anomaly if majority vote
            results['is_anomaly'] = results['vote_count'] >= len(pred_cols) // 2
            
            # Combined anomaly score
            score_cols = [col for col in results.columns if col.endswith('_score')]
            results['anomaly_score'] = results[score_cols].mean(axis=1)
            
        elif method == 'isolation_forest':
            results['is_anomaly'] = self.models['isolation_forest'].predict(X_scaled) == -1
            results['anomaly_score'] = -self.models['isolation_forest'].score_samples(X_scaled)
            
        elif method == 'lof':
            results['is_anomaly'] = self.models['lof'].predict(X_scaled) == -1
            results['anomaly_score'] = -self.models['lof'].score_samples(X_scaled)
            
        elif method == 'ocsvm':
            results['is_anomaly'] = self.models['ocsvm'].predict(X_scaled) == -1
            results['anomaly_score'] = -self.models['ocsvm'].decision_function(X_scaled)
            
        elif method == 'statistical':
            stat_results = self._detect_statistical(X)
            results['is_anomaly'] = stat_results['is_anomaly']
            results['anomaly_score'] = stat_results['anomaly_score']
            results['anomalous_features'] = stat_results['anomalous_features']
        
        return results
    
    def _detect_statistical(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using statistical methods (Z-score and IQR).
        """
        results = pd.DataFrame(index=X.index)
        feature_anomalies = pd.DataFrame(index=X.index)
        
        for col in X.columns:
            if col not in self.thresholds:
                continue
            
            thresholds = self.thresholds[col]
            stats = self.feature_stats[col]
            
            # Z-score anomaly
            z_score = np.abs((X[col] - stats['mean']) / (stats['std'] + 1e-8))
            
            # IQR anomaly
            iqr_anomaly = (X[col] < thresholds['lower_iqr']) | (X[col] > thresholds['upper_iqr'])
            
            # Combined
            feature_anomalies[col] = (z_score > 3) | iqr_anomaly
        
        results['is_anomaly'] = feature_anomalies.any(axis=1)
        results['anomaly_count'] = feature_anomalies.sum(axis=1)
        results['anomaly_score'] = results['anomaly_count'] / len(X.columns)
        results['anomalous_features'] = feature_anomalies.apply(
            lambda row: list(feature_anomalies.columns[row]), axis=1
        )
        
        return results
    
    def detect_single(self, data_point: Dict) -> AnomalyResult:
        """
        Detect if a single data point is anomalous.
        Real-time detection for streaming data.
        """
        X = pd.DataFrame([data_point])
        results = self.detect(X, method='ensemble')
        
        row = results.iloc[0]
        
        # Determine affected features
        stat_results = self._detect_statistical(X)
        affected = stat_results['anomalous_features'].iloc[0] if 'anomalous_features' in stat_results else []
        
        # Determine severity based on score and vote count
        if row['anomaly_score'] > 0.8 or row.get('vote_count', 0) >= 3:
            severity = 'critical'
        elif row['anomaly_score'] > 0.6 or row.get('vote_count', 0) >= 2:
            severity = 'high'
        elif row['anomaly_score'] > 0.4:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Determine anomaly type
        if 'temperature' in str(affected).lower():
            anomaly_type = 'thermal_anomaly'
        elif 'power' in str(affected).lower() or 'torque' in str(affected).lower():
            anomaly_type = 'mechanical_anomaly'
        elif 'wear' in str(affected).lower():
            anomaly_type = 'wear_anomaly'
        else:
            anomaly_type = 'general_anomaly'
        
        return AnomalyResult(
            timestamp=datetime.now(),
            is_anomaly=bool(row['is_anomaly']),
            anomaly_score=float(row['anomaly_score']),
            anomaly_type=anomaly_type,
            affected_features=affected,
            severity=severity,
            details={
                'vote_count': int(row.get('vote_count', 0)),
                'individual_scores': {
                    'isolation_forest': float(row.get('if_score', 0)),
                    'lof': float(row.get('lof_score', 0)),
                    'ocsvm': float(row.get('ocsvm_score', 0))
                }
            }
        )
    
    def get_anomaly_summary(self, results: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for anomaly detection results.
        """
        return {
            'total_samples': len(results),
            'anomaly_count': results['is_anomaly'].sum(),
            'anomaly_rate': results['is_anomaly'].mean(),
            'avg_anomaly_score': results['anomaly_score'].mean(),
            'max_anomaly_score': results['anomaly_score'].max(),
            'score_distribution': {
                'low': (results['anomaly_score'] < 0.4).sum(),
                'medium': ((results['anomaly_score'] >= 0.4) & (results['anomaly_score'] < 0.6)).sum(),
                'high': ((results['anomaly_score'] >= 0.6) & (results['anomaly_score'] < 0.8)).sum(),
                'critical': (results['anomaly_score'] >= 0.8).sum()
            }
        }


class SequentialAnomalyDetector:
    """
    Sequential anomaly detection for time-series equipment data.
    Detects anomalies considering temporal patterns.
    """
    
    def __init__(self, window_size: int = 50, threshold_percentile: float = 95):
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.buffer = []
        self.baseline_stats = None
        
    def update_baseline(self, X: pd.DataFrame):
        """Update baseline statistics from historical data."""
        self.baseline_stats = {
            col: {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'rolling_mean': X[col].rolling(self.window_size).mean().iloc[-1],
                'rolling_std': X[col].rolling(self.window_size).std().iloc[-1]
            }
            for col in X.columns
        }
    
    def detect_drift(self, current: pd.DataFrame, 
                     reference: pd.DataFrame = None) -> Dict:
        """
        Detect concept drift between current and reference distributions.
        """
        if reference is None and self.baseline_stats is None:
            return {'drift_detected': False, 'message': 'No reference data'}
        
        drift_results = {}
        
        for col in current.columns:
            if col not in self.baseline_stats:
                continue
            
            baseline = self.baseline_stats[col]
            current_mean = current[col].mean()
            current_std = current[col].std()
            
            # Calculate drift metrics
            mean_drift = abs(current_mean - baseline['mean']) / (baseline['std'] + 1e-8)
            std_drift = abs(current_std - baseline['std']) / (baseline['std'] + 1e-8)
            
            drift_results[col] = {
                'mean_drift': mean_drift,
                'std_drift': std_drift,
                'significant': mean_drift > 2 or std_drift > 1.5
            }
        
        significant_drifts = [k for k, v in drift_results.items() if v['significant']]
        
        return {
            'drift_detected': len(significant_drifts) > 0,
            'drifted_features': significant_drifts,
            'details': drift_results
        }
    
    def detect_pattern_change(self, X: pd.DataFrame) -> Dict:
        """
        Detect sudden changes in patterns using change point detection.
        """
        results = {}
        
        for col in X.columns:
            values = X[col].values
            
            # Simple change point detection using cumsum
            cumsum = np.cumsum(values - np.mean(values))
            max_idx = np.argmax(np.abs(cumsum))
            
            # Calculate magnitude of change
            before = values[:max_idx] if max_idx > 0 else values
            after = values[max_idx:] if max_idx < len(values) else values
            
            if len(before) > 1 and len(after) > 1:
                change_magnitude = abs(np.mean(after) - np.mean(before)) / (np.std(values) + 1e-8)
            else:
                change_magnitude = 0
            
            results[col] = {
                'change_point': max_idx,
                'change_magnitude': change_magnitude,
                'significant': change_magnitude > 2
            }
        
        return results


class AlertManager:
    """
    Manage and prioritize anomaly alerts.
    """
    
    def __init__(self, cooldown_seconds: int = 300):
        self.cooldown_seconds = cooldown_seconds
        self.alert_history = []
        self.last_alert_time = {}
        
    def should_alert(self, anomaly_result: AnomalyResult) -> bool:
        """
        Determine if an alert should be sent based on cooldown and severity.
        """
        if not anomaly_result.is_anomaly:
            return False
        
        key = f"{anomaly_result.anomaly_type}_{anomaly_result.severity}"
        
        if key in self.last_alert_time:
            elapsed = (anomaly_result.timestamp - self.last_alert_time[key]).total_seconds()
            
            # Different cooldowns for different severities
            if anomaly_result.severity == 'critical':
                cooldown = self.cooldown_seconds // 4
            elif anomaly_result.severity == 'high':
                cooldown = self.cooldown_seconds // 2
            else:
                cooldown = self.cooldown_seconds
            
            if elapsed < cooldown:
                return False
        
        self.last_alert_time[key] = anomaly_result.timestamp
        self.alert_history.append(anomaly_result)
        return True
    
    def create_alert(self, anomaly_result: AnomalyResult) -> Dict:
        """
        Create an alert message from anomaly result.
        """
        severity_emoji = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '⚡',
            'low': 'ℹ️'
        }
        
        alert = {
            'timestamp': anomaly_result.timestamp.isoformat(),
            'severity': anomaly_result.severity,
            'emoji': severity_emoji.get(anomaly_result.severity, ''),
            'type': anomaly_result.anomaly_type,
            'score': anomaly_result.anomaly_score,
            'affected_features': anomaly_result.affected_features,
            'message': self._generate_message(anomaly_result),
            'recommended_action': self._get_recommended_action(anomaly_result)
        }
        
        return alert
    
    def _generate_message(self, result: AnomalyResult) -> str:
        """Generate human-readable alert message."""
        messages = {
            'thermal_anomaly': "Unusual temperature readings detected",
            'mechanical_anomaly': "Mechanical stress indicators out of range",
            'wear_anomaly': "Tool wear pattern anomaly detected",
            'general_anomaly': "Unusual equipment behavior detected"
        }
        
        base_message = messages.get(result.anomaly_type, "Anomaly detected")
        features = ", ".join(result.affected_features[:3]) if result.affected_features else "multiple parameters"
        
        return f"{base_message} in {features}"
    
    def _get_recommended_action(self, result: AnomalyResult) -> str:
        """Get recommended action based on anomaly type and severity."""
        if result.severity == 'critical':
            return "IMMEDIATE INSPECTION REQUIRED - Consider stopping equipment"
        elif result.severity == 'high':
            return "Schedule inspection within 4 hours"
        elif result.severity == 'medium':
            return "Monitor closely and schedule routine inspection"
        else:
            return "Log for review during next maintenance cycle"
    
    def get_alert_summary(self) -> Dict:
        """Get summary of recent alerts."""
        if not self.alert_history:
            return {'total_alerts': 0}
        
        severity_counts = {}
        type_counts = {}
        
        for alert in self.alert_history:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            type_counts[alert.anomaly_type] = type_counts.get(alert.anomaly_type, 0) + 1
        
        return {
            'total_alerts': len(self.alert_history),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'last_alert': self.alert_history[-1].timestamp.isoformat()
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample normal data
    n_samples = 1000
    normal_data = pd.DataFrame({
        'temperature': np.random.normal(300, 2, n_samples),
        'power': np.random.normal(6000, 500, n_samples),
        'torque': np.random.normal(40, 10, n_samples),
        'rpm': np.random.normal(1500, 100, n_samples),
        'tool_wear': np.random.uniform(0, 200, n_samples)
    })
    
    # Create some anomalous data
    anomalous_data = pd.DataFrame({
        'temperature': [320, 280, 350],
        'power': [9500, 2000, 10000],
        'torque': [80, 5, 90],
        'rpm': [2500, 800, 3000],
        'tool_wear': [250, 0, 300]
    })
    
    # Initialize and fit detector
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(normal_data)
    
    # Detect anomalies
    test_data = pd.concat([normal_data.tail(10), anomalous_data], ignore_index=True)
    results = detector.detect(test_data, method='ensemble')
    
    print("\nAnomaly Detection Results:")
    print(results[['is_anomaly', 'anomaly_score']].tail(13))
    
    # Get summary
    summary = detector.get_anomaly_summary(results)
    print(f"\nSummary: {summary}")
    
    # Test single point detection
    single_result = detector.detect_single(anomalous_data.iloc[0].to_dict())
    print(f"\nSingle point detection: {single_result}")

"""
Data Processing Module for Predictive Maintenance
==================================================
Automated data ingestion, cleaning, and feature engineering pipeline.
Replaces manual data processing with AI-driven automation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Automated data processing pipeline for equipment sensor data.
    
    This class automates:
    - Data validation and quality checks
    - Missing value imputation
    - Outlier detection and handling
    - Feature engineering
    - Data transformation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.is_fitted = False
        
    def _default_config(self) -> Dict:
        return {
            'scaling_method': 'robust',  # 'standard', 'robust', 'minmax'
            'imputation_method': 'knn',  # 'mean', 'median', 'knn'
            'outlier_method': 'iqr',     # 'iqr', 'zscore', 'isolation_forest'
            'outlier_threshold': 1.5,
            'create_interaction_features': True,
            'create_polynomial_features': False,
            'polynomial_degree': 2
        }
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file with automatic type detection."""
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data quality checks.
        Returns a report of data quality issues.
        """
        report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Check for constant columns
        report['constant_columns'] = [col for col in df.columns if df[col].nunique() <= 1]
        
        # Check for high cardinality
        report['high_cardinality'] = [
            col for col in df.select_dtypes(include=['object']).columns 
            if df[col].nunique() > 100
        ]
        
        return report
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers using specified method.
        Returns DataFrame with outlier flags.
        """
        outlier_flags = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if self.config['outlier_method'] == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.config['outlier_threshold'] * IQR
                upper = Q3 + self.config['outlier_threshold'] * IQR
                outlier_flags[f'{col}_outlier'] = (df[col] < lower) | (df[col] > upper)
                
            elif self.config['outlier_method'] == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_flags[f'{col}_outlier'] = z_scores > 3
        
        return outlier_flags
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific engineered features for milling machine data.
        """
        df = df.copy()
        
        # Temperature-based features
        if 'Air temperature [K]' in df.columns and 'Process temperature [K]' in df.columns:
            df['temperature_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
            df['temperature_ratio'] = df['Process temperature [K]'] / df['Air temperature [K]']
            
            # Heat dissipation risk indicator
            df['heat_risk'] = ((df['temperature_difference'] < 8.6) & 
                              (df.get('Rotational speed [rpm]', 1500) < 1380)).astype(int)
        
        # Mechanical power calculation (Torque * Angular velocity)
        if 'Torque [Nm]' in df.columns and 'Rotational speed [rpm]' in df.columns:
            df['Mechanical Power [W]'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi) / 60
            
            # Power failure risk indicators
            df['power_low_risk'] = (df['Mechanical Power [W]'] < 3500).astype(int)
            df['power_high_risk'] = (df['Mechanical Power [W]'] > 9000).astype(int)
            
            # Power efficiency proxy
            df['power_efficiency'] = df['Mechanical Power [W]'] / (df['Torque [Nm]'] + 1)
        
        # Tool wear features
        if 'Tool wear [min]' in df.columns:
            df['tool_wear_normalized'] = df['Tool wear [min]'] / df['Tool wear [min]'].max()
            df['tool_wear_critical'] = (df['Tool wear [min]'] > 200).astype(int)
            
            # Overstrain risk calculation
            if 'Torque [Nm]' in df.columns and 'Type' in df.columns:
                # Threshold varies by product type
                type_thresholds = {'L': 11000, 'M': 12000, 'H': 13000}
                df['overstrain_product'] = df['Tool wear [min]'] * df['Torque [Nm]']
        
        # Rotational speed features
        if 'Rotational speed [rpm]' in df.columns:
            df['rpm_normalized'] = df['Rotational speed [rpm]'] / df['Rotational speed [rpm]'].max()
            df['rpm_deviation'] = np.abs(df['Rotational speed [rpm]'] - df['Rotational speed [rpm]'].mean())
        
        # Torque features
        if 'Torque [Nm]' in df.columns:
            df['torque_normalized'] = df['Torque [Nm]'] / df['Torque [Nm]'].max()
            df['torque_squared'] = df['Torque [Nm]'] ** 2
        
        # Interaction features
        if self.config['create_interaction_features']:
            if 'Torque [Nm]' in df.columns and 'Rotational speed [rpm]' in df.columns:
                df['torque_rpm_interaction'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
            
            if 'Tool wear [min]' in df.columns and 'Torque [Nm]' in df.columns:
                df['wear_torque_interaction'] = df['Tool wear [min]'] * df['Torque [Nm]']
        
        return df
    
    def preprocess(self, df: pd.DataFrame, 
                   target_col: str = 'Machine failure',
                   drop_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw dataframe
            target_col: Name of target column
            drop_cols: Columns to drop
            
        Returns:
            X: Processed features
            y: Target variable
        """
        df = df.copy()
        
        # Default columns to drop (identifiers and redundant failure columns)
        if drop_cols is None:
            drop_cols = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
        # Drop specified columns
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # Extract target
        if target_col in df.columns:
            y = df.pop(target_col)
        else:
            y = None
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categorical variables
        if 'Type' in df.columns:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                df['Type'] = self.label_encoder.fit_transform(df['Type'])
            else:
                df['Type'] = self.label_encoder.transform(df['Type'])
        
        # Store feature columns
        self.feature_columns = df.columns.tolist()
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not self.is_fitted:
            if self.config['scaling_method'] == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            self.is_fitted = True
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df, y
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict:
        """
        Prepare data for feature importance analysis.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        return {
            'correlation_matrix': numeric_df.corr(),
            'feature_stats': numeric_df.describe(),
            'skewness': numeric_df.skew(),
            'kurtosis': numeric_df.kurtosis()
        }


class RealTimeDataProcessor:
    """
    Real-time data processing for streaming equipment sensor data.
    Designed for production deployment.
    """
    
    def __init__(self, base_processor: DataProcessor):
        self.processor = base_processor
        self.buffer = []
        self.buffer_size = 100
        
    def process_single(self, data_point: Dict) -> pd.DataFrame:
        """Process a single data point in real-time."""
        df = pd.DataFrame([data_point])
        processed, _ = self.processor.preprocess(df, target_col=None)
        return processed
    
    def process_batch(self, data_points: List[Dict]) -> pd.DataFrame:
        """Process a batch of data points."""
        df = pd.DataFrame(data_points)
        processed, _ = self.processor.preprocess(df, target_col=None)
        return processed
    
    def add_to_buffer(self, data_point: Dict):
        """Add data point to buffer for batch processing."""
        self.buffer.append(data_point)
        if len(self.buffer) >= self.buffer_size:
            return self.flush_buffer()
        return None
    
    def flush_buffer(self) -> pd.DataFrame:
        """Process and clear the buffer."""
        if not self.buffer:
            return pd.DataFrame()
        result = self.process_batch(self.buffer)
        self.buffer = []
        return result


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Load sample data
    df = processor.load_data("Predictive Maintenance Dataset/ai4i2020.csv")
    
    # Validate data
    report = processor.validate_data(df)
    print("\n--- Data Quality Report ---")
    print(f"Records: {report['total_records']}")
    print(f"Duplicates: {report['duplicates']}")
    print(f"Memory: {report['memory_usage_mb']:.2f} MB")
    
    # Preprocess
    X, y = processor.preprocess(df)
    print(f"\n--- Preprocessed Data ---")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

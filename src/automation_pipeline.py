"""
Automated Data Processing Pipeline
===================================
End-to-end automation for equipment data processing.
Demonstrates AI-driven approach to replace manual activities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the automation pipeline."""
    input_path: str
    output_path: str
    model_path: str = "models/"
    batch_size: int = 1000
    enable_anomaly_detection: bool = True
    enable_failure_prediction: bool = True
    enable_xai: bool = True
    alert_threshold: float = 0.5
    parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class ProcessingResult:
    """Result from pipeline processing."""
    success: bool
    records_processed: int
    anomalies_detected: int
    high_risk_predictions: int
    processing_time: float
    errors: List[str] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


class AutomatedPipeline:
    """
    Automated pipeline for equipment data processing.
    
    Replaces manual data processing with AI-driven automation:
    1. Automated data ingestion and validation
    2. Intelligent feature engineering
    3. Real-time anomaly detection
    4. Predictive maintenance scoring
    5. Automated alerting and reporting
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processor = None
        self.anomaly_detector = None
        self.model = None
        self.explainer = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing automated pipeline...")
        
        # Initialize data processor
        from src.data_processor import DataProcessor
        self.processor = DataProcessor()
        
        # Initialize anomaly detector
        if self.config.enable_anomaly_detection:
            from src.anomaly_detection import AnomalyDetector
            self.anomaly_detector = AnomalyDetector()
        
        # Load trained model
        if self.config.enable_failure_prediction:
            self._load_model()
        
        # Initialize explainer
        if self.config.enable_xai and self.model is not None:
            from src.explainability import ModelExplainer
            # Will be initialized with training data
        
        self.is_initialized = True
        logger.info("✓ Pipeline initialized successfully")
    
    def _load_model(self):
        """Load the trained prediction model."""
        import joblib
        
        model_file = os.path.join(self.config.model_path, "best_model.joblib")
        
        if os.path.exists(model_file):
            self.model = joblib.load(model_file)
            logger.info(f"✓ Model loaded from {model_file}")
        else:
            logger.warning(f"Model file not found: {model_file}")
    
    def process_file(self, filepath: str) -> ProcessingResult:
        """
        Process a single data file through the pipeline.
        
        Args:
            filepath: Path to the input data file
            
        Returns:
            ProcessingResult with processing summary
        """
        start_time = datetime.now()
        errors = []
        
        try:
            # Step 1: Load and validate data
            logger.info(f"Processing file: {filepath}")
            df = self.processor.load_data(filepath)
            
            validation_report = self.processor.validate_data(df)
            logger.info(f"  Loaded {validation_report['total_records']} records")
            
            # Step 2: Preprocess and engineer features
            X, y = self.processor.preprocess(df)
            logger.info(f"  Engineered {X.shape[1]} features")
            
            # Step 3: Anomaly detection
            anomalies_detected = 0
            if self.config.enable_anomaly_detection and self.anomaly_detector is not None:
                if not self.anomaly_detector.is_fitted:
                    self.anomaly_detector.fit(X)
                
                anomaly_results = self.anomaly_detector.detect(X)
                anomalies_detected = anomaly_results['is_anomaly'].sum()
                logger.info(f"  Detected {anomalies_detected} anomalies")
            
            # Step 4: Failure prediction
            high_risk_count = 0
            predictions = None
            if self.config.enable_failure_prediction and self.model is not None:
                predictions = self.model.predict_proba(X)[:, 1]
                high_risk_count = (predictions > self.config.alert_threshold).sum()
                logger.info(f"  {high_risk_count} high-risk predictions")
            
            # Step 5: Generate alerts for high-risk items
            if high_risk_count > 0:
                self._generate_alerts(X, predictions)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                records_processed=len(df),
                anomalies_detected=anomalies_detected,
                high_risk_predictions=high_risk_count,
                processing_time=processing_time,
                summary={
                    'file': filepath,
                    'features': X.shape[1],
                    'anomaly_rate': anomalies_detected / len(df) if len(df) > 0 else 0,
                    'high_risk_rate': high_risk_count / len(df) if len(df) > 0 else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return ProcessingResult(
                success=False,
                records_processed=0,
                anomalies_detected=0,
                high_risk_predictions=0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                errors=[str(e)]
            )
    
    def process_batch(self, filepaths: List[str]) -> List[ProcessingResult]:
        """
        Process multiple files in parallel.
        
        Args:
            filepaths: List of file paths to process
            
        Returns:
            List of ProcessingResult for each file
        """
        results = []
        
        if self.config.parallel_processing:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_file, fp): fp 
                    for fp in filepaths
                }
                
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {filepath}: {e}")
                        results.append(ProcessingResult(
                            success=False,
                            records_processed=0,
                            anomalies_detected=0,
                            high_risk_predictions=0,
                            processing_time=0,
                            errors=[str(e)]
                        ))
        else:
            for filepath in filepaths:
                results.append(self.process_file(filepath))
        
        return results
    
    def _generate_alerts(self, X: pd.DataFrame, predictions: np.ndarray):
        """Generate alerts for high-risk predictions."""
        high_risk_mask = predictions > self.config.alert_threshold
        high_risk_indices = np.where(high_risk_mask)[0]
        
        alerts = []
        for idx in high_risk_indices[:10]:  # Limit to top 10
            alert = {
                'index': int(idx),
                'probability': float(predictions[idx]),
                'timestamp': datetime.now().isoformat(),
                'features': X.iloc[idx].to_dict() if hasattr(X, 'iloc') else {}
            }
            alerts.append(alert)
        
        # Save alerts
        alert_file = os.path.join(self.config.output_path, 'alerts.json')
        os.makedirs(os.path.dirname(alert_file), exist_ok=True)
        
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)
        
        logger.info(f"  Generated {len(alerts)} alerts")
    
    def run_continuous(self, watch_path: str, interval: int = 60):
        """
        Run pipeline continuously, watching for new files.
        
        Args:
            watch_path: Directory to watch for new files
            interval: Check interval in seconds
        """
        import time
        
        processed_files = set()
        logger.info(f"Starting continuous monitoring of {watch_path}")
        
        while True:
            try:
                # Check for new files
                for filename in os.listdir(watch_path):
                    if filename.endswith('.csv'):
                        filepath = os.path.join(watch_path, filename)
                        
                        # Check if already processed
                        file_hash = self._get_file_hash(filepath)
                        if file_hash in processed_files:
                            continue
                        
                        # Process new file
                        result = self.process_file(filepath)
                        processed_files.add(file_hash)
                        
                        logger.info(f"Processed {filename}: {result.records_processed} records")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping continuous monitoring")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(interval)
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get hash of file for change detection."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()


class DataQualityChecker:
    """
    Automated data quality checking.
    Replaces manual data validation with intelligent checks.
    """
    
    def __init__(self):
        self.checks = []
        self.results = {}
        
    def add_check(self, name: str, check_func: Callable, 
                  severity: str = 'warning'):
        """Add a quality check."""
        self.checks.append({
            'name': name,
            'func': check_func,
            'severity': severity
        })
    
    def run_checks(self, df: pd.DataFrame) -> Dict:
        """Run all quality checks on data."""
        results = {
            'passed': [],
            'warnings': [],
            'errors': [],
            'summary': {}
        }
        
        for check in self.checks:
            try:
                passed, message = check['func'](df)
                
                result = {
                    'name': check['name'],
                    'passed': passed,
                    'message': message,
                    'severity': check['severity']
                }
                
                if passed:
                    results['passed'].append(result)
                elif check['severity'] == 'error':
                    results['errors'].append(result)
                else:
                    results['warnings'].append(result)
                    
            except Exception as e:
                results['errors'].append({
                    'name': check['name'],
                    'passed': False,
                    'message': str(e),
                    'severity': 'error'
                })
        
        results['summary'] = {
            'total_checks': len(self.checks),
            'passed': len(results['passed']),
            'warnings': len(results['warnings']),
            'errors': len(results['errors']),
            'pass_rate': len(results['passed']) / len(self.checks) if self.checks else 0
        }
        
        return results
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.05):
        """Check for excessive missing values."""
        missing_rate = df.isnull().sum().sum() / df.size
        passed = missing_rate < threshold
        message = f"Missing value rate: {missing_rate:.2%} (threshold: {threshold:.0%})"
        return passed, message
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame, threshold: float = 0.01):
        """Check for duplicate records."""
        dup_rate = df.duplicated().sum() / len(df)
        passed = dup_rate < threshold
        message = f"Duplicate rate: {dup_rate:.2%} (threshold: {threshold:.0%})"
        return passed, message
    
    @staticmethod
    def check_value_ranges(df: pd.DataFrame, expected_ranges: Dict):
        """Check if values are within expected ranges."""
        issues = []
        for col, (min_val, max_val) in expected_ranges.items():
            if col not in df.columns:
                continue
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range > 0:
                issues.append(f"{col}: {out_of_range} values out of range")
        
        passed = len(issues) == 0
        message = "; ".join(issues) if issues else "All values within expected ranges"
        return passed, message


class ReportGenerator:
    """
    Automated report generation.
    Creates comprehensive reports from pipeline results.
    """
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
    
    def generate_summary_report(self, results: List[ProcessingResult]) -> str:
        """Generate summary report from processing results."""
        report = []
        report.append("=" * 60)
        report.append("AUTOMATED PROCESSING SUMMARY REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # Overall statistics
        total_records = sum(r.records_processed for r in results)
        total_anomalies = sum(r.anomalies_detected for r in results)
        total_high_risk = sum(r.high_risk_predictions for r in results)
        total_time = sum(r.processing_time for r in results)
        success_count = sum(1 for r in results if r.success)
        
        report.append("\n## Processing Summary")
        report.append(f"  Files Processed: {len(results)}")
        report.append(f"  Successful: {success_count}/{len(results)}")
        report.append(f"  Total Records: {total_records:,}")
        report.append(f"  Total Processing Time: {total_time:.2f}s")
        report.append(f"  Average Throughput: {total_records/total_time:.0f} records/sec")
        
        report.append("\n## Detection Summary")
        report.append(f"  Anomalies Detected: {total_anomalies:,}")
        report.append(f"  Anomaly Rate: {total_anomalies/total_records*100:.2f}%" if total_records > 0 else "N/A")
        report.append(f"  High-Risk Predictions: {total_high_risk:,}")
        report.append(f"  High-Risk Rate: {total_high_risk/total_records*100:.2f}%" if total_records > 0 else "N/A")
        
        # Errors
        all_errors = [e for r in results for e in r.errors]
        if all_errors:
            report.append("\n## Errors")
            for error in all_errors[:10]:
                report.append(f"  - {error}")
        
        report.append("\n" + "=" * 60)
        
        # Save report
        report_text = "\n".join(report)
        report_file = os.path.join(
            self.output_path, 
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def generate_json_report(self, results: List[ProcessingResult]) -> Dict:
        """Generate JSON report for programmatic consumption."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'files_processed': len(results),
                'successful': sum(1 for r in results if r.success),
                'failed': sum(1 for r in results if not r.success),
                'total_records': sum(r.records_processed for r in results),
                'total_anomalies': sum(r.anomalies_detected for r in results),
                'total_high_risk': sum(r.high_risk_predictions for r in results),
                'total_processing_time': sum(r.processing_time for r in results)
            },
            'file_results': [
                {
                    'success': r.success,
                    'records': r.records_processed,
                    'anomalies': r.anomalies_detected,
                    'high_risk': r.high_risk_predictions,
                    'time': r.processing_time,
                    'errors': r.errors
                }
                for r in results
            ]
        }
        
        # Save JSON report
        json_file = os.path.join(
            self.output_path,
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


if __name__ == "__main__":
    # Example usage
    config = PipelineConfig(
        input_path="Predictive Maintenance Dataset/",
        output_path="output/",
        model_path="models/"
    )
    
    pipeline = AutomatedPipeline(config)
    
    # Initialize with dummy data processor (actual initialization requires models)
    print("Pipeline configuration:")
    print(f"  Input: {config.input_path}")
    print(f"  Output: {config.output_path}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Anomaly detection: {config.enable_anomaly_detection}")
    print(f"  Failure prediction: {config.enable_failure_prediction}")
    print(f"  Parallel processing: {config.parallel_processing}")

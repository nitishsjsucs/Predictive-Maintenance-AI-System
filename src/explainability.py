"""
Explainable AI (XAI) Module for Predictive Maintenance
=======================================================
Provides interpretable insights into model predictions using SHAP, LIME, and other XAI techniques.
Critical for understanding WHY a machine is predicted to fail.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")


class ModelExplainer:
    """
    Comprehensive model explainability using multiple XAI techniques.
    
    Provides:
    - Global feature importance
    - Local (instance-level) explanations
    - SHAP values analysis
    - LIME explanations
    - Feature interaction analysis
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame, feature_names: List[str] = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained ML model
            X_train: Training data for background distribution
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.shap_values = None
        
    def setup_shap(self, explainer_type: str = 'auto', background_samples: int = 100):
        """
        Setup SHAP explainer.
        
        Args:
            explainer_type: 'tree', 'kernel', 'linear', or 'auto'
            background_samples: Number of background samples for kernel explainer
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available")
            return
        
        print("Setting up SHAP explainer...")
        
        # Sample background data
        if len(self.X_train) > background_samples:
            background = shap.sample(self.X_train, background_samples)
        else:
            background = self.X_train
        
        if explainer_type == 'auto':
            # Auto-detect best explainer type
            model_name = type(self.model).__name__.lower()
            
            if 'forest' in model_name or 'tree' in model_name or 'boost' in model_name:
                explainer_type = 'tree'
            elif 'linear' in model_name or 'logistic' in model_name:
                explainer_type = 'linear'
            else:
                explainer_type = 'kernel'
        
        try:
            if explainer_type == 'tree':
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif explainer_type == 'linear':
                self.shap_explainer = shap.LinearExplainer(self.model, background)
            else:
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') 
                    else self.model.predict,
                    background
                )
            print(f"✓ SHAP {explainer_type} explainer ready")
        except Exception as e:
            print(f"Error setting up SHAP: {e}")
            # Fallback to kernel explainer
            self.shap_explainer = shap.KernelExplainer(
                lambda x: self.model.predict_proba(x)[:, 1] if hasattr(self.model, 'predict_proba')
                else self.model.predict(x),
                background
            )
    
    def setup_lime(self, mode: str = 'classification'):
        """Setup LIME explainer."""
        if not LIME_AVAILABLE:
            print("LIME not available")
            return
        
        print("Setting up LIME explainer...")
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['No Failure', 'Failure'],
            mode=mode,
            discretize_continuous=True
        )
        print("✓ LIME explainer ready")
    
    def compute_shap_values(self, X: pd.DataFrame, max_samples: int = 500) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Data to explain
            max_samples: Maximum samples to compute (for performance)
        """
        if self.shap_explainer is None:
            self.setup_shap()
        
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=42)
        
        print(f"Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.shap_explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Take positive class
        
        return self.shap_values
    
    def get_global_importance(self, X: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.
        """
        if self.shap_values is None:
            if X is not None:
                self.compute_shap_values(X)
            else:
                self.compute_shap_values(self.X_train)
        
        # Calculate mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })
        
        return df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    def explain_instance_shap(self, instance: pd.DataFrame, 
                               plot: bool = True) -> Dict:
        """
        Explain a single prediction using SHAP.
        
        Args:
            instance: Single row DataFrame
            plot: Whether to show force plot
        """
        if self.shap_explainer is None:
            self.setup_shap()
        
        shap_values = self.shap_explainer.shap_values(instance)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            prob = self.model.predict_proba(instance)[0, 1]
        else:
            prob = self.model.predict(instance)[0]
        
        # Create explanation dictionary
        explanation = {
            'prediction_probability': prob,
            'feature_contributions': dict(zip(self.feature_names, shap_values[0])),
            'top_positive_features': [],
            'top_negative_features': []
        }
        
        # Sort contributions
        contributions = list(zip(self.feature_names, shap_values[0]))
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        explanation['top_positive_features'] = [
            {'feature': f, 'contribution': c} 
            for f, c in contributions if c > 0
        ][:5]
        
        explanation['top_negative_features'] = [
            {'feature': f, 'contribution': c} 
            for f, c in contributions if c < 0
        ][-5:]
        
        if plot and SHAP_AVAILABLE:
            shap.force_plot(
                self.shap_explainer.expected_value if not isinstance(self.shap_explainer.expected_value, list)
                else self.shap_explainer.expected_value[1],
                shap_values[0],
                instance.iloc[0],
                feature_names=self.feature_names,
                matplotlib=True,
                show=True
            )
        
        return explanation
    
    def explain_instance_lime(self, instance: pd.DataFrame, 
                               num_features: int = 10) -> Dict:
        """
        Explain a single prediction using LIME.
        """
        if self.lime_explainer is None:
            self.setup_lime()
        
        # Get LIME explanation
        exp = self.lime_explainer.explain_instance(
            instance.values[0],
            self.model.predict_proba if hasattr(self.model, 'predict_proba') 
            else lambda x: np.column_stack([1-self.model.predict(x), self.model.predict(x)]),
            num_features=num_features
        )
        
        # Parse explanation
        explanation = {
            'prediction': exp.predict_proba[1] if len(exp.predict_proba) > 1 else exp.predict_proba[0],
            'feature_rules': exp.as_list(),
            'local_prediction': exp.local_pred[0] if hasattr(exp, 'local_pred') else None
        }
        
        return explanation
    
    def plot_summary(self, X: pd.DataFrame = None, plot_type: str = 'bar',
                     max_display: int = 15, save_path: str = None):
        """
        Create SHAP summary plot.
        
        Args:
            X: Data to explain (uses training data if None)
            plot_type: 'bar', 'dot', or 'violin'
            max_display: Maximum features to display
            save_path: Path to save the plot
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available")
            return
        
        if X is None:
            X = self.X_train
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(self.shap_values, X, plot_type='bar', 
                            max_display=max_display, show=False)
        elif plot_type == 'dot':
            shap.summary_plot(self.shap_values, X, 
                            max_display=max_display, show=False)
        elif plot_type == 'violin':
            shap.summary_plot(self.shap_values, X, plot_type='violin',
                            max_display=max_display, show=False)
        
        plt.title('SHAP Feature Importance', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_dependence(self, feature: str, X: pd.DataFrame = None,
                        interaction_feature: str = 'auto',
                        save_path: str = None):
        """
        Create SHAP dependence plot for a specific feature.
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available")
            return
        
        if X is None:
            X = self.X_train
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, self.shap_values, X,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.title(f'SHAP Dependence: {feature}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def get_feature_interactions(self, X: pd.DataFrame = None,
                                  top_n: int = 10) -> pd.DataFrame:
        """
        Analyze feature interactions using SHAP interaction values.
        """
        if not SHAP_AVAILABLE:
            return pd.DataFrame()
        
        if X is None:
            X = self.X_train.sample(min(100, len(self.X_train)))
        
        if not hasattr(self.shap_explainer, 'shap_interaction_values'):
            print("Interaction values not available for this explainer type")
            return pd.DataFrame()
        
        try:
            interaction_values = self.shap_explainer.shap_interaction_values(X)
            
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[1]
            
            # Calculate mean absolute interaction
            mean_interactions = np.abs(interaction_values).mean(axis=0)
            
            # Create interaction matrix
            interaction_df = pd.DataFrame(
                mean_interactions,
                columns=self.feature_names,
                index=self.feature_names
            )
            
            return interaction_df
            
        except Exception as e:
            print(f"Error computing interactions: {e}")
            return pd.DataFrame()
    
    def generate_explanation_report(self, X: pd.DataFrame, 
                                    instance_idx: int = None) -> str:
        """
        Generate a human-readable explanation report.
        """
        report = []
        report.append("=" * 60)
        report.append("PREDICTIVE MAINTENANCE - MODEL EXPLANATION REPORT")
        report.append("=" * 60)
        
        # Global importance
        importance = self.get_global_importance(X)
        report.append("\n## Global Feature Importance (Top 10)")
        report.append("-" * 40)
        for _, row in importance.head(10).iterrows():
            bar = "█" * int(row['Importance'] * 20 / importance['Importance'].max())
            report.append(f"{row['Feature']:30} {bar} {row['Importance']:.4f}")
        
        # Instance explanation if provided
        if instance_idx is not None:
            instance = X.iloc[[instance_idx]]
            report.append(f"\n## Instance Explanation (Index: {instance_idx})")
            report.append("-" * 40)
            
            explanation = self.explain_instance_shap(instance, plot=False)
            report.append(f"Failure Probability: {explanation['prediction_probability']:.2%}")
            
            report.append("\nTop factors INCREASING failure risk:")
            for item in explanation['top_positive_features']:
                report.append(f"  + {item['feature']}: {item['contribution']:+.4f}")
            
            report.append("\nTop factors DECREASING failure risk:")
            for item in explanation['top_negative_features']:
                report.append(f"  - {item['feature']}: {item['contribution']:+.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class FailureRootCauseAnalyzer:
    """
    Analyze root causes of predicted failures using XAI.
    Maps feature contributions to actionable maintenance insights.
    """
    
    # Domain knowledge mapping: features to maintenance actions
    FEATURE_TO_ACTION = {
        'temperature_difference': {
            'high': 'Check cooling system efficiency',
            'low': 'Inspect heat dissipation mechanism'
        },
        'Mechanical Power [W]': {
            'high': 'Reduce load or check for mechanical stress',
            'low': 'Inspect power transmission system'
        },
        'Tool wear [min]': {
            'high': 'Schedule tool replacement',
            'critical': 'Immediate tool replacement required'
        },
        'Torque [Nm]': {
            'high': 'Check for material buildup or obstruction',
            'low': 'Verify motor and drive system'
        },
        'Rotational speed [rpm]': {
            'high': 'Reduce speed or check bearings',
            'low': 'Inspect motor and control system'
        },
        'heat_risk': {
            'high': 'Critical: Improve ventilation and reduce speed'
        },
        'power_low_risk': {
            'high': 'Check power supply and motor efficiency'
        },
        'power_high_risk': {
            'high': 'Reduce load immediately'
        },
        'overstrain_product': {
            'high': 'Reduce torque or replace worn tool'
        }
    }
    
    def __init__(self, explainer: ModelExplainer):
        self.explainer = explainer
    
    def analyze(self, instance: pd.DataFrame, 
                feature_values: pd.DataFrame = None) -> Dict:
        """
        Analyze root cause and provide maintenance recommendations.
        """
        # Get SHAP explanation
        explanation = self.explainer.explain_instance_shap(instance, plot=False)
        
        # Analyze contributing factors
        recommendations = []
        risk_factors = []
        
        for item in explanation['top_positive_features']:
            feature = item['feature']
            contribution = item['contribution']
            
            if feature in self.FEATURE_TO_ACTION:
                action_map = self.FEATURE_TO_ACTION[feature]
                
                # Determine severity
                if contribution > 0.5:
                    severity = 'critical'
                elif contribution > 0.2:
                    severity = 'high'
                else:
                    severity = 'moderate'
                
                action = action_map.get(severity, action_map.get('high', 'Monitor closely'))
                
                recommendations.append({
                    'feature': feature,
                    'contribution': contribution,
                    'severity': severity,
                    'action': action
                })
                
                risk_factors.append({
                    'factor': feature,
                    'impact': contribution,
                    'current_value': instance[feature].values[0] if feature in instance.columns else None
                })
        
        return {
            'failure_probability': explanation['prediction_probability'],
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'urgency': 'HIGH' if explanation['prediction_probability'] > 0.7 else 
                      'MEDIUM' if explanation['prediction_probability'] > 0.4 else 'LOW'
        }
    
    def generate_maintenance_ticket(self, analysis: Dict, 
                                    machine_id: str = 'UNKNOWN') -> str:
        """
        Generate a maintenance ticket from analysis.
        """
        ticket = []
        ticket.append("=" * 60)
        ticket.append("MAINTENANCE TICKET - AUTO-GENERATED")
        ticket.append("=" * 60)
        ticket.append(f"\nMachine ID: {machine_id}")
        ticket.append(f"Urgency: {analysis['urgency']}")
        ticket.append(f"Failure Probability: {analysis['failure_probability']:.1%}")
        
        ticket.append("\n--- RISK FACTORS ---")
        for factor in analysis['risk_factors']:
            ticket.append(f"  • {factor['factor']}: Impact = {factor['impact']:.3f}")
            if factor['current_value'] is not None:
                ticket.append(f"    Current Value: {factor['current_value']:.2f}")
        
        ticket.append("\n--- RECOMMENDED ACTIONS ---")
        for i, rec in enumerate(analysis['recommendations'], 1):
            ticket.append(f"  {i}. [{rec['severity'].upper()}] {rec['action']}")
            ticket.append(f"     Related to: {rec['feature']}")
        
        ticket.append("\n" + "=" * 60)
        
        return "\n".join(ticket)


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=500, n_features=8, random_state=42)
    feature_names = ['temp_diff', 'power', 'tool_wear', 'torque', 
                     'rpm', 'heat_risk', 'power_risk', 'strain']
    X = pd.DataFrame(X, columns=feature_names)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = ModelExplainer(model, X, feature_names)
    explainer.setup_shap(explainer_type='tree')
    
    # Get global importance
    importance = explainer.get_global_importance(X.head(100))
    print("\nGlobal Feature Importance:")
    print(importance)
    
    # Explain single instance
    print("\nExplaining instance 0:")
    exp = explainer.explain_instance_shap(X.head(1), plot=False)
    print(f"Prediction: {exp['prediction_probability']:.2%}")
    print("Top positive factors:", exp['top_positive_features'][:3])

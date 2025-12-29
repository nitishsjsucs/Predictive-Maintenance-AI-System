"""
Interactive Dashboard for Predictive Maintenance
=================================================
Streamlit-based real-time monitoring dashboard.
Provides visualization, predictions, and actionable insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor
from src.anomaly_detection import AnomalyDetector, AlertManager


# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-critical {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #1a1a2e;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(filepath):
    """Load and cache data."""
    return pd.read_csv(filepath)


@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    model_path = "models/"
    
    if os.path.exists(model_path):
        for filename in os.listdir(model_path):
            if filename.endswith('.joblib'):
                model_name = filename.replace('.joblib', '')
                models[model_name] = joblib.load(os.path.join(model_path, filename))
    
    return models


def create_gauge_chart(value, title, thresholds=[30, 70]):
    """Create a gauge chart for metrics."""
    if value < thresholds[0]:
        color = "green"
    elif value < thresholds[1]:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, thresholds[0]], 'color': 'lightgreen'},
                {'range': [thresholds[0], thresholds[1]], 'color': 'lightyellow'},
                {'range': [thresholds[1], 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_time_series_chart(df, columns, title):
    """Create interactive time series chart."""
    fig = make_subplots(rows=len(columns), cols=1, 
                        shared_xaxes=True,
                        subplot_titles=columns,
                        vertical_spacing=0.05)
    
    colors = px.colors.qualitative.Set2
    
    for i, col in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=df.index if 'timestamp' not in df.columns else df['timestamp'],
                y=df[col],
                name=col,
                line=dict(color=colors[i % len(colors)], width=2),
                mode='lines'
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        height=200 * len(columns),
        title_text=title,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_failure_distribution_chart(df):
    """Create failure type distribution chart."""
    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    available_cols = [col for col in failure_cols if col in df.columns]
    
    if not available_cols:
        return None
    
    failure_counts = df[available_cols].sum()
    
    fig = go.Figure(data=[
        go.Bar(
            x=failure_counts.index,
            y=failure_counts.values,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            text=failure_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Failure Type Distribution",
        xaxis_title="Failure Type",
        yaxis_title="Count",
        height=400
    )
    
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        width=800
    )
    
    return fig


def create_feature_importance_chart(importance_dict):
    """Create feature importance chart."""
    df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Importance'],
            y=df['Feature'],
            orientation='h',
            marker_color='steelblue'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400
    )
    
    return fig


def render_sidebar():
    """Render sidebar with controls and info."""
    st.sidebar.markdown("## Control Panel")
    
    st.sidebar.markdown("""
    <div class="sidebar-info">
        <strong>AI4I Predictive Maintenance</strong><br>
        AI-powered equipment monitoring and failure prediction system.
    </div>
    """, unsafe_allow_html=True)
    
    # Data upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Equipment Data (CSV)",
        type=['csv'],
        help="Upload sensor data for analysis"
    )
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "Gradient Boosting", "XGBoost", "Ensemble"],
        index=0
    )
    
    # Threshold settings
    st.sidebar.markdown("### Alert Thresholds")
    failure_threshold = st.sidebar.slider(
        "Failure Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    anomaly_threshold = st.sidebar.slider(
        "Anomaly Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05
    )
    
    # Refresh settings
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    return {
        'uploaded_file': uploaded_file,
        'model': model_option,
        'failure_threshold': failure_threshold,
        'anomaly_threshold': anomaly_threshold,
        'auto_refresh': auto_refresh
    }


def render_overview_tab(df):
    """Render overview dashboard tab."""
    st.markdown("## Equipment Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_records = len(df)
    failure_rate = df['Machine failure'].mean() * 100 if 'Machine failure' in df.columns else 0
    avg_tool_wear = df['Tool wear [min]'].mean() if 'Tool wear [min]' in df.columns else 0
    avg_power = df['Mechanical Power [W]'].mean() if 'Mechanical Power [W]' in df.columns else 0
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{total_records:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Failure Rate",
            value=f"{failure_rate:.2f}%",
            delta=f"{failure_rate - 3.39:.2f}%" if failure_rate else None,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Avg Tool Wear",
            value=f"{avg_tool_wear:.1f} min",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Avg Power",
            value=f"{avg_power:.0f} W",
            delta=None
        )
    
    # Gauges
    st.markdown("### Equipment Health Indicators")
    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
    
    with gauge_col1:
        health_score = max(0, 100 - failure_rate * 10)
        st.plotly_chart(
            create_gauge_chart(health_score, "Overall Health", [60, 80]),
            use_container_width=True
        )
    
    with gauge_col2:
        wear_score = (avg_tool_wear / 253) * 100 if avg_tool_wear else 0
        st.plotly_chart(
            create_gauge_chart(wear_score, "Tool Wear Level", [50, 80]),
            use_container_width=True
        )
    
    with gauge_col3:
        temp_diff = df['Process temperature [K]'].mean() - df['Air temperature [K]'].mean() if all(col in df.columns for col in ['Process temperature [K]', 'Air temperature [K]']) else 10
        temp_score = min(100, (temp_diff / 12) * 100)
        st.plotly_chart(
            create_gauge_chart(temp_score, "Thermal Status", [40, 70]),
            use_container_width=True
        )


def render_analysis_tab(df):
    """Render data analysis tab."""
    st.markdown("## Data Analysis")
    
    # Distribution of product types
    if 'Type' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            type_counts = df['Type'].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Product Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Machine failure' in df.columns:
                fig = px.histogram(
                    df,
                    x='Type',
                    color='Machine failure',
                    barmode='group',
                    title="Failures by Product Type",
                    labels={'Machine failure': 'Failure Status'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.markdown("### Feature Distributions")
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if available_cols:
        selected_feature = st.selectbox("Select Feature", available_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x=selected_feature,
                nbins=50,
                title=f"Distribution of {selected_feature}",
                color_discrete_sequence=['steelblue']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df,
                y=selected_feature,
                title=f"Box Plot of {selected_feature}",
                color_discrete_sequence=['steelblue']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### Correlation Analysis")
    st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)


def render_prediction_tab(df, settings):
    """Render prediction tab."""
    st.markdown("## Failure Prediction")
    
    # Input form for single prediction
    st.markdown("### Single Equipment Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        product_type = st.selectbox("Product Type", ['L', 'M', 'H'])
        air_temp = st.number_input("Air Temperature [K]", 
                                   min_value=290.0, max_value=310.0, value=300.0, step=0.1)
        process_temp = st.number_input("Process Temperature [K]", 
                                        min_value=300.0, max_value=320.0, value=310.0, step=0.1)
    
    with col2:
        rpm = st.number_input("Rotational Speed [rpm]", 
                              min_value=1000, max_value=3000, value=1500, step=10)
        torque = st.number_input("Torque [Nm]", 
                                  min_value=0.0, max_value=100.0, value=40.0, step=0.5)
    
    with col3:
        tool_wear = st.number_input("Tool Wear [min]", 
                                     min_value=0, max_value=300, value=100, step=5)
    
    if st.button("Predict Failure Risk", type="primary"):
        # Calculate derived features
        temp_diff = process_temp - air_temp
        power = (torque * rpm * 2 * np.pi) / 60
        
        # Create prediction display
        st.markdown("---")
        
        # Simulated prediction (replace with actual model prediction)
        failure_prob = 0.0
        
        # Rule-based risk calculation (domain knowledge)
        if temp_diff < 8.6 and rpm < 1380:
            failure_prob += 0.3  # Heat dissipation risk
        if power < 3500 or power > 9000:
            failure_prob += 0.25  # Power failure risk
        if tool_wear > 200:
            failure_prob += 0.35  # Tool wear risk
        
        type_threshold = {'L': 11000, 'M': 12000, 'H': 13000}
        if tool_wear * torque > type_threshold.get(product_type, 12000):
            failure_prob += 0.2  # Overstrain risk
        
        failure_prob = min(failure_prob, 0.99)
        
        # Display results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.plotly_chart(
                create_gauge_chart(failure_prob * 100, "Failure Probability", [30, 70]),
                use_container_width=True
            )
        
        with result_col2:
            if failure_prob < 0.3:
                status = "LOW RISK"
                status_class = "status-healthy"
                recommendation = "Equipment operating within normal parameters. Continue standard monitoring."
            elif failure_prob < 0.7:
                status = "MEDIUM RISK"
                status_class = "status-warning"
                recommendation = "Elevated risk detected. Schedule preventive maintenance within 48 hours."
            else:
                status = "HIGH RISK"
                status_class = "status-critical"
                recommendation = "CRITICAL: High failure probability. Immediate inspection recommended."
            
            st.markdown(f"""
            <div style="padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
                <h3 class="{status_class}">{status}</h3>
                <p><strong>Failure Probability:</strong> {failure_prob:.1%}</p>
                <p><strong>Recommendation:</strong> {recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk factors
            st.markdown("#### Risk Factors")
            if temp_diff < 8.6:
                st.warning("Low temperature difference - Heat dissipation concern")
            if power < 3500:
                st.warning("Low mechanical power output")
            if power > 9000:
                st.error("High mechanical power - Potential overload")
            if tool_wear > 200:
                st.error("Critical tool wear level - Replace tool soon")


def render_anomaly_tab(df):
    """Render anomaly detection tab."""
    st.markdown("## Anomaly Detection")
    
    # Prepare features for anomaly detection
    feature_cols = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if not available_cols:
        st.warning("Required columns not found for anomaly detection.")
        return
    
    X = df[available_cols].copy()
    
    # Simple anomaly detection using Z-score
    z_scores = np.abs((X - X.mean()) / X.std())
    anomaly_mask = (z_scores > 3).any(axis=1)
    
    anomaly_count = anomaly_mask.sum()
    anomaly_rate = anomaly_count / len(df) * 100
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Anomalies Detected", f"{anomaly_count:,}")
    
    with col2:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    
    with col3:
        status = "Normal" if anomaly_rate < 5 else "Elevated" if anomaly_rate < 10 else "Critical"
        st.metric("System Status", status)
    
    # Anomaly visualization
    st.markdown("### Anomaly Distribution")
    
    if anomaly_count > 0:
        anomaly_df = df[anomaly_mask].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df,
                x='Rotational speed [rpm]' if 'Rotational speed [rpm]' in df.columns else available_cols[0],
                y='Torque [Nm]' if 'Torque [Nm]' in df.columns else available_cols[1],
                color=anomaly_mask.astype(str),
                title="Anomaly Detection: RPM vs Torque",
                labels={'color': 'Is Anomaly'},
                color_discrete_map={'True': 'red', 'False': 'blue'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature-wise anomaly count
            feature_anomalies = (z_scores > 3).sum()
            fig = px.bar(
                x=feature_anomalies.index,
                y=feature_anomalies.values,
                title="Anomalies by Feature",
                labels={'x': 'Feature', 'y': 'Anomaly Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent anomalies table
        st.markdown("### Recent Anomalies")
        st.dataframe(
            anomaly_df.tail(10),
            use_container_width=True
        )
    else:
        st.success("No significant anomalies detected in the current dataset.")


def main():
    """Main dashboard application."""
    # Header
    st.markdown("""
    <div class="main-header">
        Predictive Maintenance Dashboard
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    settings = render_sidebar()
    
    # Load data
    if settings['uploaded_file'] is not None:
        df = pd.read_csv(settings['uploaded_file'])
        st.success(f"Loaded {len(df)} records from uploaded file")
    else:
        # Try to load default data
        default_path = "Predictive Maintenance Dataset/ai4i2020.csv"
        if os.path.exists(default_path):
            df = load_data(default_path)
        else:
            st.info("Please upload equipment data using the sidebar to get started.")
            st.markdown("""
            ### Welcome to the Predictive Maintenance Dashboard
            
            This AI-powered dashboard provides:
            - **Real-time Equipment Monitoring**
            - **Failure Prediction** using Machine Learning
            - **Anomaly Detection** for early warning
            - **Data Analysis** and visualization
            - **Actionable Recommendations**
            
            Upload your equipment sensor data (CSV format) to begin analysis.
            """)
            return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", 
        "Analysis", 
        "Prediction", 
        "Anomalies"
    ])
    
    with tab1:
        render_overview_tab(df)
    
    with tab2:
        render_analysis_tab(df)
    
    with tab3:
        render_prediction_tab(df, settings)
    
    with tab4:
        render_anomaly_tab(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.8rem;">
        AI-Powered Predictive Maintenance System | 
        Equipment Monitoring and Failure Prediction
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

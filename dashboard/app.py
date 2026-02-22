import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime
import os

FASTAPI_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MLOps Drift Monitor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Model Drift Monitoring Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Connection
    st.subheader("API Connection")
    api_url = st.text_input("API URL", value=FASTAPI_URL)
    
    # Test connection
    if st.button("🔄 Test Connection"):
        try:
            response = requests.get(f"{api_url}/healthz", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connected to API")
            else:
                st.error(f"❌ API returned status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot connect to API: {str(e)}")
    
    st.markdown("---")
    
    # Dashboard Settings
    st.subheader("Dashboard Settings")
    refresh_rate = st.slider("Refresh Rate (seconds)", min_value=1, max_value=10, value=5)
    max_history = st.slider("Max History Points", min_value=10, max_value=100, value=50)
    
    st.markdown("---")
    
    # Info
    st.subheader("ℹ️ About")
    st.markdown("""
    This dashboard monitors concept drift in real-time.
    - **Drift Score**: 0-1 scale
    - **Threshold**: 0.15 (15%)
    - **Alert**: Triggers when score > threshold
    """)
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.timestamps = []
        st.rerun()

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []
if 'last_alert' not in st.session_state:
    st.session_state.last_alert = False

# Fetch drift score from API
def fetch_drift():
    try:
        resp = requests.get(f"{api_url}/drift_score", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error: {resp.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the server is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def fetch_stats():
    try:
        resp = requests.get(f"{api_url}/stats", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None

# Main content area - Create placeholders
col1, col2, col3 = st.columns(3)
with col1:
    drift_placeholder = st.empty()
with col2:
    samples_placeholder = st.empty()
with col3:
    status_placeholder = st.empty()

chart_placeholder = st.empty()
stats_placeholder = st.empty()
alert_placeholder = st.empty()

# Auto-refresh loop
while True:
    # Fetch data
    data = fetch_drift()
    stats = fetch_stats()
    
    if data:
        score = data['drift_score']
        alert = data['alert']
        threshold = data['threshold']
        recent_samples = data['recent_samples']
        baseline_samples = data['baseline_samples']
        
        # Update history
        st.session_state.history.append(score)
        st.session_state.timestamps.append(datetime.now())
        
        # Keep only last N points
        if len(st.session_state.history) > max_history:
            st.session_state.history = st.session_state.history[-max_history:]
            st.session_state.timestamps = st.session_state.timestamps[-max_history:]
        
        # Display metrics
        with col1:
            drift_placeholder.metric(
                "Current Drift Score",
                f"{score:.3f}",
                delta=f"{score - threshold:.3f} vs threshold",
                delta_color="inverse"
            )
        
        with col2:
            samples_placeholder.metric(
                "Total Samples",
                recent_samples,
                delta=f"Baseline: {baseline_samples}"
            )
        
        with col3:
            status = "⚠️ DRIFT" if alert else "✅ NORMAL"
            status_placeholder.metric("System Status", status)
        
        # Alert
        if alert and not st.session_state.last_alert:
            alert_placeholder.error("🚨 **DRIFT DETECTED!** Check Discord for alerts.")
            st.session_state.last_alert = True
        elif not alert:
            alert_placeholder.success("✅ System normal - No drift detected")
            st.session_state.last_alert = False
        
        # Plot drift history
        if len(st.session_state.history) > 0:
            df = pd.DataFrame({
                'Time': st.session_state.timestamps,
                'Drift Score': st.session_state.history
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Time'],
                y=df['Drift Score'],
                mode='lines+markers',
                name='Drift Score',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Add threshold line
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({threshold*100}%)",
                annotation_position="top right"
            )
            
            # Add fill for area above threshold
            fig.add_hrect(
                y0=threshold,
                y1=1,
                line_width=0,
                fillcolor="red",
                opacity=0.1
            )
            
            fig.update_layout(
                title="Drift Score Over Time",
                xaxis_title="Time",
                yaxis_title="Drift Score",
                hovermode='x unified',
                showlegend=True,
                height=400
            )
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Display statistics
        if stats:
            st.subheader("📈 Prediction Statistics")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Total Predictions", stats.get('total_predictions', 0))
            
            with col_b:
                if stats.get('recent_distribution'):
                    dist = stats['recent_distribution']
                    st.write("**Class Distribution:**")
                    for class_name, count in dist.items():
                        st.write(f"- Class {class_name}: {count}")
            
            with col_c:
                if stats.get('recent_distribution'):
                    dist = stats['recent_distribution']
                    # Simple bar chart of distribution
                    dist_df = pd.DataFrame({
                        'Class': list(dist.keys()),
                        'Count': list(dist.values())
                    })
                    st.bar_chart(dist_df.set_index('Class'))
    
    else:
        # Show connection error
        drift_placeholder.error("⚠️ Waiting for API connection...")
    
    # Wait before next refresh
    time.sleep(refresh_rate)
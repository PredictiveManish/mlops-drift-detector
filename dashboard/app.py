import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import time

FASTAPI_URL = "https://your-fastapi-app.onrender.com"  # change after deployment

st.set_page_config(layout="wide")
st.title("Model Drift Monitor")

# Fetch drift score periodically
def fetch_drift():
    try:
        resp = requests.get(f"{FASTAPI_URL}/drift_score")
        return resp.json()
    except:
        return None

drift_placeholder = st.empty()
chart_placeholder = st.empty()
alert_placeholder = st.empty()

# We'll store drift history in session state
if 'history' not in st.session_state:
    st.session_state.history = []

while True:
    data = fetch_drift()
    if data:
        score = data['drift_score']
        alert = data['alert']
        st.session_state.history.append(score)
        
        # Current score
        drift_placeholder.metric("Current Drift Score", f"{score:.3f}")
        
        # Alert
        if alert:
            alert_placeholder.error("🚨 Drift threshold exceeded! Check Discord.")
        else:
            alert_placeholder.success("✅ No drift detected.")
        
        # Plot history
        df = pd.DataFrame({'drift_score': st.session_state.history})
        fig = go.Figure(data=go.Scatter(y=df['drift_score'], mode='lines+markers'))
        fig.add_hline(y=0.15, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(title="Drift Score Over Time", xaxis_title="Check #", yaxis_title="Score")
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    time.sleep(5)   # refresh every 5 seconds
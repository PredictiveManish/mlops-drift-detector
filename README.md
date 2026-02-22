# MLOps Model Drift Monitoring System

[![CI](https://github.com/predictivemanish/mlops-drift-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/predictivemanish/mlops-drift-detector/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready MLOps system that detects concept drift in machine learning models through real-time statistical performance comparison. Built with FastAPI, Streamlit, and scikit-learn.

## 📋 Features

- **Real-time Drift Detection**: Statistical comparison using PSI (Population Stability Index) and KS tests
- **Fast Inference**: <500ms prediction latency
- **Configurable Alerts**: 15% threshold with Discord webhook integration
- **Interactive Dashboard**: Streamlit dashboard with Plotly visualizations
- **Async Logging**: Non-blocking prediction logging to SQLite
- **Docker Support**: Containerized deployment
- **CI/CD**: Automated testing via GitHub Actions
- **Zero-cost Deployment**: Ready for Render (API) and Hugging Face Spaces (Dashboard)

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Client      │───▶│ FastAPI      │───▶│ SQLite      │
│ (curl/app)  │    │ Inference    │    │ Database    │
└─────────────┘    └──────────────┘    └─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ Drift        │    │ Streamlit   │    │ Discord      │
│ Detector     │───▶│ Dashboard   │───▶│ Webhook      │
└──────────────┘    └─────────────┘    └──────────────┘

```


## 🔧 Local Installation
#### 1. Clone the Repository
Open your terminal/command prompt and run:
```bash
git clone https://github.com/YOUR_USERNAME/mlops-drift-detector.git
cd mlops-drift-detector

" 💡 Pro Tip: If you don't see "Open in Terminal" when right-clicking a folder, click 2-3 times until it appears. "

```

#### 2. Create Virtual Environment using venv or conda (as you want)

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Train and save model
```bash
cd app
python model.py
cd ..
```

#### 5. Run servcies
- Open two separate terminals
1. terminal1: FastAPI Server:
```bash
uvicorn app.main:app --reload --port 8000
```

2. Terminal 2: Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

#### 6. Access Applications
API: http://localhost:8000
API Docs: http://localhost:8000/docs
Dashboard: http://localhost:8501


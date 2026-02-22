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
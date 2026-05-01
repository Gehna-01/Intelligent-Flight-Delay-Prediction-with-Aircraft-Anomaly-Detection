# ✈️ Intelligent Flight Delay Prediction System

### End-to-End Machine Learning Pipeline with Aircraft Sensor Intelligence

---

## 🚀 Overview

This project builds a **machine learning system** to predict flight delays by combining:

* ✈️ Flight operations data
* 🌦️ Weather conditions
* ⚙️ Aircraft sensor telemetry (CMAPSS dataset)

Unlike traditional systems, this approach integrates **operational + environmental + mechanical signals** to improve prediction reliability.

---

## System Interface Preview

The system provides a real-time dashboard for monitoring flight schedules, delay predictions, and aircraft health signals.

<img width="2048" height="1102" alt="image" src="https://github.com/user-attachments/assets/d1e47cf6-3935-450a-90c6-6a5cacee402e" />

---

## 🎯 Problem Statement

Flight delays are driven by multiple factors:

* Weather volatility
* Air traffic congestion
* Aircraft health

Most models rely only on historical schedules and weather.

> This system extends prediction capability by incorporating **sensor-based anomaly detection**, enabling deeper insight into aircraft-level risks.

---

## 🧠 System Architecture

The system is divided into two core pipelines:

### 1️⃣ Flight Delay Prediction Engine

* Supervised learning (Random Forest, XGBoost)
* Predicts delay probability using structured features

### 2️⃣ Aircraft Anomaly Detection Engine

* Unsupervised learning (Isolation Forest / Autoencoder)
* Detects abnormal sensor behavior

> Sensor anomalies act as **early risk indicators** for potential delays.

---

## Architecture Breakdown

### Flight Delay Prediction Engine

Supervised learning pipeline that processes flight and weather features to estimate delay probability.

<img width="1933" height="1321" alt="image" src="https://github.com/user-attachments/assets/bca33150-fb43-4dc2-ade6-b9f2d8ff7687" />


### Aircraft Sensor Anomaly Detection

Unsupervised pipeline that analyzes engine telemetry to detect abnormal behavior patterns.

<img width="1881" height="1075" alt="image" src="https://github.com/user-attachments/assets/81723f62-1aed-4a40-97c9-84a385b89805" />

---

## ⚙️ Key Features

* End-to-end ML pipeline (data → training → evaluation → visualization)
* Feature engineering across multiple data sources
* Integration of flight, weather, and sensor data
* Unsupervised anomaly detection on telemetry
* Standard evaluation metrics (F1, ROC-AUC, etc.)
* Interactive Streamlit dashboard

---

## 🗂️ Project Structure

```bash
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned & engineered data
│
├── notebooks/                  # EDA & experimentation
│
├── src/                        # Core ML pipelines
│   ├── preprocess.py
│   ├── train_delay_model.py
│   ├── anomaly_detection.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/                     # Saved models
├── app/                        # Streamlit dashboard
├── outputs/                    # Reports & visualizations
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
git clone <repo_url>
cd flight-delay-prediction
pip install -r requirements.txt
```

---

## 📊 Dataset

### Source

* Dataset Folder: `(https://drive.google.com/drive/folders/1trIm62kNPJnZXy9Lpib0qaWD27naDGhy?usp=sharing)`

### Setup

```bash
data/
├── raw/
│   ├── flight_data.csv
│   ├── weather_data.csv
│   └── cmapss_sensor_data.csv
```

* Place all files inside `data/raw/`
* File names must remain unchanged
* Processed data will be generated automatically

---

## 🔄 Usage

## Prediction & Diagnostics

The system provides detailed diagnostics for each flight, including delay probability and sensor anomaly scores.

<img width="1863" height="910" alt="image" src="https://github.com/user-attachments/assets/990218e8-0270-4f9a-8e81-33fb8e1b6c77" />

---


## Feature Inputs & Telemetry

The model utilizes structured weather data and high-dimensional sensor telemetry as input features.

<img width="1711" height="1200" alt="image" src="https://github.com/user-attachments/assets/e521a377-f3fd-4bf3-ad6c-772f13303e20" />

<img width="1593" height="1102" alt="image" src="https://github.com/user-attachments/assets/706508f6-c260-49a7-92cc-576a7990e148" />

<img width="1593" height="1102" alt="image" src="https://github.com/user-attachments/assets/30234818-a3b2-4647-a1ff-3372be1c7717" />

---

### 1. Data Preprocessing

```python
from src.preprocess import preprocess_pipeline

preprocess_pipeline(
    input_path='data/raw/flight_data.csv',
    output_path='data/processed/flight_processed.csv'
)
```

---

### 2. Model Training

```python
from src.train_delay_model import train_pipeline
import pandas as pd

df = pd.read_csv('data/processed/flight_processed.csv')

X = df.drop('delay', axis=1)
y = df['delay']

model = train_pipeline(X, y)
```

---

### 3. Anomaly Detection

```python
from src.anomaly_detection import detect_anomalies
import pandas as pd

sensor_data = pd.read_csv('data/raw/cmapss_sensor_data.csv')

predictions, detector = detect_anomalies(
    sensor_data.values,
    contamination=0.05
)
```

---

### 4. Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Evaluation

### Classification Metrics

* Accuracy
* Precision / Recall
* F1 Score
* ROC-AUC

### Anomaly Detection

* Outlier score distribution
* Sensitivity tuning via contamination

---

## 🧩 Design Decisions

* **Tree-based models** → strong performance on structured data
* **Isolation Forest** → efficient for high-dimensional anomaly detection
* **Modular architecture** → scalable and maintainable

---

## ⚠️ Limitations

* CMAPSS dataset is simulated
* Batch-based pipeline (no real-time streaming)
* Limited model interpretability

---

## 🚀 Future Improvements

* Real-time prediction pipeline
* Time-series models (LSTM for sensor data)
* Model explainability (SHAP / LIME)
* API deployment (FastAPI)
* Integration with airline systems

---

## 📜 License

MIT License

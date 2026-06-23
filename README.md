# 🏦 Bank Customer Churn Prediction using Machine Learning

## 🌐 Live Demo

[![Live Demo](https://img.shields.io/badge/Streamlit-Live_App-red?style=for-the-badge&logo=streamlit)](https://bankcustomerchurnprediction-a7u8icoui9mkfsgtjhmxdm.streamlit.app/)

---

## Overview

Built an end-to-end Machine Learning solution to predict customer churn in the banking sector.

The project leverages customer demographic and financial information to predict whether a customer is likely to leave the bank. The solution includes data preprocessing, feature engineering, model training, model serialization, and deployment through an interactive Streamlit web application.

---

## Business Problem

Customer retention is significantly more cost-effective than customer acquisition.

Banks lose revenue when valuable customers leave their services. This project helps financial institutions proactively identify customers at risk of churn, enabling targeted retention strategies and improved customer satisfaction.

---

## Technical Implementation

### Machine Learning Model

- Gradient Boosting Classifier
- Binary Classification Problem
- Probability-based Churn Prediction
- Hyperparameter Tuning
- Model Evaluation using Classification Metrics

### Data Engineering

- Label Encoding (Gender)
- One-Hot Encoding (Geography)
- Feature Scaling using StandardScaler
- Train-Test Split for Evaluation
- Model Serialization using Joblib and Pickle

### Model Evaluation Metrics

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- ROC-AUC Score

---

## Deployment

The trained model is deployed using Streamlit, providing a user-friendly interface for real-time customer churn prediction.

The deployment ensures:

- Consistent preprocessing pipeline
- Fast and scalable inference
- Interactive user experience
- Real-time churn probability estimation

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Gradient Boost
- Streamlit
- Joblib
- Pickle

---

## Repository Structure

```text
BANK_CUSTOMER_CHURN/
│
├── app.py                      # Streamlit application
├── best_model.pkl              # Trained XGBoost model
├── scaler.pkl                  # StandardScaler object
├── label_encoder_gender.pkl    # Gender encoder
├── onehot_encoder_geo.pkl      # Geography encoder
├── training.ipynb              # Model training notebook
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## Key Features

- End-to-End Machine Learning Pipeline
- Customer Churn Prediction
- Data Preprocessing & Feature Engineering
- Model Serialization for Deployment
- Interactive Streamlit Dashboard
- Real-world Banking Use Case
- Probability-based Risk Assessment

---

## Input Features

The model uses the following customer attributes:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card Status
- Active Membership Status
- Estimated Salary

---

## Output

The application provides:

- Churn Prediction (Likely to Churn / Not Likely to Churn)
- Churn Probability Score
- Interactive Prediction Results

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/InduNallagopu/BANK_CUSTOMER_CHURN.git
```

### 2. Navigate to Project Directory

```bash
cd BANK_CUSTOMER_CHURN
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## Future Improvements

- Customer Retention Recommendation System
- Feature Importance Visualization
- Interactive Analytics Dashboard
- Cloud-Based Model Monitoring
- Automated Retraining Pipeline

---

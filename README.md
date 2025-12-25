#  Disease Prediction System using Machine Learning

A Streamlit web application that predicts diseases based on patient symptoms, demographic information, and environmental factors using machine learning.

---

##  Project Overview

This project implements a complete machine learning–based disease prediction system.  
It analyzes **patient symptoms**, **demographic information**, and **environmental factors** such as temperature, humidity, and wind speed to predict one of **11 disease classes**.

The system is designed for **educational and research purposes** and demonstrates how machine learning can support preliminary medical screening.

---

##  Objectives

- Build a multi-class disease classification model
- Analyze the impact of symptoms and environmental factors on disease prediction
- Deploy the trained model as a user-friendly web application
- Demonstrate end-to-end machine learning deployment using Streamlit

---

##  Machine Learning Models

The following models were evaluated:

- Logistic Regression (Baseline Model)
- Support Vector Machine (SVM)
- Random Forest Classifier ✅ (Final Model)

**Final Model Used:** Random Forest  
**Test Accuracy:** **98.29%**

Random Forest was selected due to:
- High predictive accuracy
- Robustness to noise
- Ability to handle mixed feature types
- Feature importance interpretation

---

##  Dataset Description

- **Initial dataset size:** 5,200 records  
- **After cleaning:** 4,981 records  
- **Target variable:** Disease (multi-class classification – 11 classes)

### Feature Categories:
- **Demographic Features:**  
  - Age  
  - Gender  

- **Environmental Features:**  
  - Temperature (°C)  
  - Humidity (%)  
  - Wind Speed (km/h)  

- **Symptom Features:**  
  - 44 binary symptom indicators (0 = absent, 1 = present)

---

##  Data Preprocessing

- Duplicate records removed
- Missing values handled
- Feature scaling applied using **StandardScaler**
- Label encoding applied to disease classes
- Stratified train-test split used to maintain class balance

---

##  Streamlit Web Application Features

- Multi-page interface:
  - Home
  - About
  - Disease Prediction
  - Model Information
- Interactive symptom selection
- Environmental and demographic input sliders
- Disease prediction with confidence score
- Probability distribution visualization
- Medical disclaimer for ethical use

---

##  Deployment

The application is deployed using:

- **GitHub** (Version control)
- **Streamlit Community Cloud** (Web hosting)

This allows the application to be accessed via a public URL without local installation.

---

## 📦 Project Structure


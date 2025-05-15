# Multi-Class Classification of Chronic Cough using Machine Learning

This project presents an **AI-powered system** to classify the **severity of chronic cough** into three categories — **normal, moderate, and severe** — using a combination of **acoustic features from cough sounds** and **biometric metadata** (age, height, weight, BMI). The solution leverages classical and deep learning models to offer a **non-invasive, objective**, and **scalable diagnostic tool** for respiratory health monitoring.

## 📌 Project Highlights

- **Objective**: Eliminate subjectivity in cough evaluation by using audio-based machine learning techniques.
- **Inputs**: Cough audio (recorded via MEMS microphone) + user metadata (age, height, weight, BMI).
- **Outputs**: Severity level — `Normal (1)`, `Moderate (2)`, `Severe (3)`.
- **Models Used**:
  - Random Forest (baseline)
  - Convolutional Neural Network (CNN)
  - Histogram-based Gradient Boosting (HistGBM - best performance)

## 🧠 Key Features

- **Acoustic Feature Extraction**:
  - MFCCs (Mel-Frequency Cepstral Coefficients)
  - RMS Energy
  - Zero Crossing Rate (ZCR)
  - Spectral Entropy
- **Biometric Context**:
  - Age, Height, Weight, BMI
- **Data Processing Pipeline**:
  - Audio preprocessing and noise filtering
  - Feature normalization and scaling
  - Train-test split (80/20) with 5-fold cross-validation

## 📊 Results

- **Best performing model**: `HistGBM`
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Optimized through**: Randomized search for hyperparameters

## 🏗️ System Architecture

```plaintext
 Cough Audio + User Metadata
         │
         ▼
Feature Extraction (MFCCs, ZCR, etc.)
         │
         ▼
   Preprocessing & Scaling
         │
         ▼
     ML Model
         │
         ▼
 Cough Severity Classification
````

## 📱 Companion App

An Android application has been developed to:

* Record cough sounds
* Collect user metadata
* Run inferences (via backend or on-device)
* Visualize classification results
* Store data securely via cloud

## 🧪 Potential Applications

* Remote patient monitoring
* Clinical decision support
* Telemedicine platforms
* Resource-limited healthcare setups


## 👨‍💻 Authors

* Uday Khunti
* Darshit Desai
* Prachi Rohit
* Kishor Upla
  *Department of Electronics, SVNIT Surat*

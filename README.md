# Deep_Learning_E_Commerce_Customer_Satisfaction_Score_Prediction
E_Commerce_Customer_Satisfaction_Score_Prediction using Deep_Learning

[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-ANN-blue.svg)]()
[![E-Commerce](https://img.shields.io/badge/E--Commerce-CSAT-brightgreen.svg)]()
[![Built With](https://img.shields.io/badge/Built%20With-Python%20%7C%20Keras%20%7C%20Flask-orange)]()

---

## 🚀 Overview

**DeepCSAT: E-Commerce Customer Satisfaction Score Prediction** harnesses the power of Deep Learning to accurately forecast customer satisfaction (CSAT) scores. By understanding user interactions and feedback in real-time, this project empowers e-commerce businesses with actionable insights, driving enhanced service quality, customer retention, and business growth.

---

## 📈 Project Goal

- **Objective:**  
  To develop a Deep Learning model (ANN) that predicts CSAT scores from customer support interactions on the "Shopzilla" e-commerce platform.
- **Business Impact:**  
  Real-time satisfaction monitoring for improved service delivery and customer loyalty.

---

## 🏗️ Project Steps

1. **Data Preparation:**  
   Data cleaning, handling missing values, and preprocessing for modeling.
2. **Feature Engineering:**  
   Identifying and extracting features most predictive of CSAT scores.
3. **Model Development:**  
   Designing, training, and optimizing an Artificial Neural Network.
4. **Evaluation:**  
   Assessing model performance using relevant metrics and validation.
5. **Insight Generation:**  
   Analyzing predictions to discover trends and guide improvements.
6. **Deployment:**  
   Flask-based local deployment for interactive predictions and live demonstrations.

---

## 📦 Dataset

**Source:** "Shopzilla" e-commerce support interactions  
**Period:** 1 month  
**Size:** ~19.2MB (`eCommerce_Customer_support_data.csv`)

### **Features:**
The dataset includes one month of customer service interactions at "Shopzilla," with:

- **Unique id**: Primary record identifier (integer)
- **Channel name**: Customer service channel (string)
- **Category/Sub-category**: Interaction type (string)
- **Customer Remarks**: Direct feedback (string)
- **Order id/Order date time/Issue reported/Issue responded/Survey response date**: Temporal & order references (various datetime)
- **Customer city**: Location (string)
- **Product category**: Product classification (string)
- **Item price**: Transaction value (float)
- **Connected handling time**: Interaction length (float)
- **Agent name/Supervisor/Manager/Tenure Bucket/Shift**: Human resource metadata (strings)
- **CSAT Score**: Target variable (integer)

---

## 🧠 Technologies Used

- **Python**
- **Pandas, Numpy** (Data processing)
- **Keras, TensorFlow** (Deep Learning models)
- **Matplotlib, Seaborn** (Exploratory Data Analysis and Visualization)
- **Flask** (Local Model Deployment)

---

## Goal

Predict Customer Satisfaction (CSAT) scores from e-commerce customer interactions using Deep Learning Artificial Neural Networks (ANNs).

---

## 📝 Overview

**DeepCSAT** leverages deep learning to predict CSAT scores in the e-commerce domain. By analyzing customer interactions and feedback, the project enables businesses to proactively monitor and improve customer satisfaction, driving retention and growth. The pipeline integrates robust data handling, advanced neural modeling, and intuitive deployment for actionable business outcomes.

---

## 🔍 Project Background

Customer satisfaction is a cornerstone for repeat business, loyalty, and positive word-of-mouth. Traditional approaches rely on slow, subjective surveys. Deep learning makes it possible to **predict CSAT in real time**, offering a detailed, data-driven perspective to spot gaps and elevate service quality.


---

## ✅ Model Artifacts

Stored in the root project directory:

- `csat_model.h5` – Trained Keras deep learning model
- `scaler.pkl` – Feature scaler used during training
- `features.pkl` – List of engineered and selected features
- `eCommerce_Customer_support_data.csv` – Full processed dataset

---

## 🚀 Deployment

- **Streamlit + Flask App (`app.py`)**:  
  Launches a lightweight UI for CSAT prediction, exposing an easy-to-use web interface.

---

## 🗂️ Key Files

| File                                                    | Purpose                              |
|---------------------------------------------------------|--------------------------------------|
| DeepCSAT_E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb | Main Jupyter notebook with all experiments and EDA |
| csat_model.h5                                           | Trained DNN model artifact           |
| app.py                                                  | Hybrid Streamlit/Flask prediction app|
| scaler.pkl, features.pkl                                 | Data preprocessing objects           |
| eCommerce_Customer_support_data.csv                     | Cleaned input dataset                |


---

## ✨ Results & Insights

- **Model Performance:**  
  Achieves high predictive accuracy on the hold-out set.
---

## 💡 Future Work

- Enhance model with advanced NLP for customer remarks.
- Integrate live data streams for ongoing CSAT monitoring.
- Deploy on cloud for enterprise scalability.

---

## 🤝 Contribution

PRs and suggestions welcome! Please fork the repo and submit pull requests.

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙋 Author

Lokesh Todi

> *Empowering e-commerce with actionable real-time customer satisfaction insights using Deep Learning.*


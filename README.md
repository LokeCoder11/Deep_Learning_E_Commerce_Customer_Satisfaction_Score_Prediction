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
- `Unique id` : Record identifier *(int)*
- `Channel name` : Service channel *(str)*
- `Category`, `Sub-category` : Interaction details *(str)*
- `Customer Remarks` : Customer feedback *(str)*
- `Order id`, `Order date time` : Order details *(int, datetime)*
- `Issue reported at`, `Issue responded`, `Survey response date` : Timestamps *(datetime)*
- `Customer city`, `Product category` : Location and item *(str)*
- `Item price`, `Connected handling time` : Numerical features *(float)*
- `Agent name`, `Supervisor`, `Manager` : Staff involved *(str)*
- `Tenure Bucket`, `Agent Shift` : Agent characteristics *(str)*
- `CSAT Score` : Satisfaction score (target) *(int)*

---

## 🧠 Technologies Used

- **Python**
- **Pandas, Numpy** (Data processing)
- **Keras, TensorFlow** (Deep Learning models)
- **Matplotlib, Seaborn** (Exploratory Data Analysis and Visualization)
- **Flask** (Local Model Deployment)

---

## 🎯 How to Run

1. **Clone the Repository**
    ```
    git clone https://github.com/YOUR_USERNAME/E_Commerce_Customer_Satisfaction_Score_Prediction.git
    cd E_Commerce_Customer_Satisfaction_Score_Prediction
    ```
2. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```
3. **Explore and Train**
   - Jupyter Notebooks:  
     - `E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb` – Full pipeline  
     - `flask_eCommerce_Customer_Satisfaction_Score_Prediction.ipynb` – Deployment prep
    ```
4. **Model Predictions**
    - Access the web app on [localhost:5000](http://localhost:5000) to input data and get CSAT predictions.
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

> *Empowering e-commerce with actionable real-time customer satisfaction insights using Deep Learning.*


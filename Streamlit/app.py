import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier

# Loaders
def load_model():
    return keras.models.load_model("csat_model.h5")

def load_scaler():
    return joblib.load('scaler.pkl')

def load_feature_list():
    return joblib.load('features.pkl')

# Preprocessing
def preprocess_new_data(data, features, numerical_features):
    empty_df = pd.DataFrame()
    for col in features:
        if col not in data.columns:
            empty_df[col] = 0
        elif col in numerical_features:
            empty_df[col] = data[col]
        else:
            empty_df[col] = 1
    return empty_df

def rename_column(df, numerical_col):
    for col in df.columns:
        if col not in numerical_col:
            first_val = df[col].iloc[0]
            df.rename(columns={col: f"{col}_{first_val}"}, inplace=True)
    return df

# UI
st.set_page_config(page_title="CSAT Prediction APP")
st.header("Customer Satisfaction Prediction System")
st.subheader("Input Features for Prediction")

# Inputs
channel_name = st.selectbox("Channel Name", ["Email", "Inbound", "Outcall"])
category = st.selectbox("Category", ['Product Queries', 'Order Related', 'Returns', 'Cancellation', 'Shopzilla Related', 'Payments related', 'Refund Related', 'Feedback', 'Offers & Cashback', 'Onboarding related', 'Others', 'App/website'])
sub_category = st.selectbox("Sub-category",['Life Insurance', 'Product Specific Information', 'Installation/demo', 'Reverse Pickup Enquiry', 'Not Needed', 'Fraudulent User', 'Exchange / Replacement', 'Missing', 'General Enquiry', 'Return request', 'Delayed', 'Service Centres Related', 'Payment related Queries', 'Order status enquiry', 'Return cancellation', 'Unable to track', 'Seller Cancelled Order', 'Wrong', 'Invoice request', 'Priority delivery', 'Refund Related Issues', 'Signup Issues', 'Online Payment Issues', 'Technician Visit', 'UnProfessional Behaviour', 'Damaged', 'Product related Issues', 'Refund Enquiry', 'Customer Requested Modifications', 'Instant discount', 'Card/EMI', 'Shopzila Premium Related', 'Account updation', 'COD Refund Details', 'Seller onboarding', 'Order Verification', 'Other Cashback', 'Call disconnected', 'Wallet related', 'PayLater related', 'Call back request', 'Other Account Related Issues', 'App/website Related', 'Affiliate Offers', 'Issues with Shopzilla App', 'Billing Related', 'Warranty related', 'Others', 'e-Gift Voucher', 'Shopzilla Rewards', 'Unable to Login', 'Non Order related', 'Service Center - Service Denial', 'Payment pending', 'Policy Related', 'Self-Help', 'Commission related'])
order_date_time = st.text_input("Order Date Time (e.g. 28-07-2023 10:52:00)")
issue_reported_at = st.text_input("Issue Reported At (e.g. 28-07-2023 10:52:00)")
issue_responded = st.text_input("Issue Responded (e.g. 28-07-2023 11:05:00)")
customer_city = st.text_input("Customer City")
product_category = st.selectbox("Product Category", ['LifeStyle', 'Electronics', 'Mobile', 'Home Appliences', 'Furniture', 'Home', 'Books & General merchandise', 'GiftCard', 'Affiliates'])
item_price = st.number_input("Item Price", min_value=0.0, step=0.01)
connected_handling_time = st.number_input("Connected Handling Time (seconds)", min_value=0.0, step=0.01)
agent_name = st.text_input("Agent Name")
supervisor = st.selectbox("Supervisor",['Mason Gupta', 'Dylan Kim', 'Jackson Park', 'Olivia Wang', 'Austin Johnson', 'Emma Park', 'Aiden Patel', 'Evelyn Kimura', 'Nathan Patel', 'Amelia Tanaka', 'Harper Wong', 'Zoe Yamamoto', 'Scarlett Chen', 'Sophia Sato', 'Wyatt Kim', 'Logan Lee', 'Mia Patel', 'William Park', 'Emily Yamashita', 'Madison Kim', 'Noah Patel', 'Oliver Nguyen', 'Elijah Yamaguchi', 'Layla Taniguchi', 'Isabella Wong', 'Carter Park', 'Jacob Sato', 'Ethan Tan', 'Mia Yamamoto', 'Brayden Wong', 'Ava Wong', 'Landon Tanaka', 'Lucas Singh', 'Charlotte Suzuki', 'Abigail Suzuki', 'Ethan Nakamura', 'Olivia Suzuki', 'Alexander Tanaka', 'Lily Chen', 'Sophia Chen'])
manager = st.selectbox("Manager", ['Jennifer Nguyen', 'Michael Lee', 'William Kim', 'John Smith', 'Olivia Tan', 'Emily Chen'])
tenure_bucket = st.selectbox("Tenure Bucket", ['On Job Training', '>90', '0-30', '31-60', '61-90'])
agent_shift = st.selectbox("Agent Shift", ['Morning', 'Evening', 'Split', 'Afternoon', 'Night'])
survey_response_date = st.text_input("Survey Response Date (e.g. 01-Aug-23)")

# Prediction
if st.button("Predict CSAT Score"):
    try:
        # Create DataFrame
        new_data = pd.DataFrame({
            'channel_name': [channel_name],
            'category': [category],
            'Sub-category': [sub_category],
            'order_date_time': [order_date_time],
            'Issue_reported at': [issue_reported_at],
            'issue_responded': [issue_responded],
            'Customer_City': [customer_city],
            'Product_category': [product_category],
            'Item_price': [item_price],
            'connected_handling_time': [connected_handling_time],
            'Agent_name': [agent_name],
            'Supervisor': [supervisor],
            'Manager': [manager],
            'Tenure Bucket': [tenure_bucket],
            'Agent Shift': [agent_shift],
            'Survey_response_Date': [survey_response_date]
        })

        # Parse dates with automatic format detection
        new_data['Issue_reported at'] = pd.to_datetime(new_data['Issue_reported at'], dayfirst=True, errors='coerce')
        new_data['issue_responded'] = pd.to_datetime(new_data['issue_responded'], dayfirst=True, errors='coerce')
        new_data['order_date_time'] = pd.to_datetime(new_data['order_date_time'], dayfirst=True, errors='coerce')
        new_data['Survey_response_Date'] = pd.to_datetime(new_data['Survey_response_Date'], dayfirst=True, errors='coerce')

        # Drop rows with bad datetime formats
        if new_data[['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']].isnull().any().any():
            st.error("Please ensure all date fields are in the correct format and try again.")
        else:
            # Feature engineering
            new_data['Response_Time_seconds'] = (new_data['issue_responded'] - new_data['Issue_reported at']).dt.total_seconds()
            new_data['day_number_order_date'] = new_data['order_date_time'].dt.day
            new_data['day_number_response_date'] = new_data['Survey_response_Date'].dt.day
            new_data['weekday_num_response_date'] = new_data['Survey_response_Date'].dt.weekday + 1
            new_data = new_data.drop(columns=['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date'])

            numerical_features = ['Item_price', 'connected_handling_time', 'Response_Time_seconds',
                                'day_number_order_date', 'day_number_response_date',
                                'weekday_num_response_date']

            # Rename columns
            new_data = rename_column(new_data, numerical_features)

            # Load scaler and features
            scaler = load_scaler()
            sorted_features = load_feature_list()

            # Preprocess
            new_data1 = preprocess_new_data(new_data, sorted_features, numerical_features)
            new_data1[numerical_features] = scaler.transform(new_data1[numerical_features])
            X_test_array = new_data1.values.astype(np.float32)

            # Predict
            keras_model = load_model()
            predictions = keras_model.predict(X_test_array)
            pred_classes = np.argmax(predictions, axis=1)


            # Output
            st.write("Prediction Results")
            st.write(f"The Predicted Customer Satisfaction Score is **{int(pred_classes)+1}**")
            st.write("All Prediction Probabilities:")
            st.write(predictions)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
import streamlit as st
import pandas as pd
import joblib


model = joblib.load('Churn_prediction_model.pkl')


st.set_page_config(page_title="Intelli_serve_india", page_icon="🛍️")
st.title("🛍️ Customer Churn Prediction")
st.write("Enter customer details below to predict whether the customer will churn or not.")


col1, col2 = st.columns(2)

with col1:
    contract_type = st.selectbox("Contract Type:", ["Annualy", "Quaterly", "Monthly"])
    days_inactive = st.slider("Days Since Last Activity:", min_value=0, max_value=30, value=10)
    tickets_raised = st.number_input("Complaint Tickets Raised:", min_value=0, max_value=8, value=1)
    satisfaction_score = st.slider("Satisfaction Score (0 to 5):", min_value=0, max_value=5, value=4)


with col2:
    tenure_months = st.number_input("Tenure (Months):", min_value=0, max_value=60, value=12)
    monthly_spend = st.number_input("Monthly Spend (₹):", min_value=1000, max_value=8000, value=3500)
    usage_drop = st.slider("Usage Drop (-1.0 to 1.0):", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)


contract_monthly = 1 if contract_type == "Monthly" else 0
contract_quaterly = 1 if contract_type == "Quaterly" else 0

if st.button("Predict Churn"):
    
    input_df = pd.DataFrame([{
        'Tenure_months': tenure_months,
        'Days_Since_Last_Activity': days_inactive,
        'Monthly_Spend': monthly_spend,
        'Complaint_Ticket_Raised': tickets_raised,
        'Satisfaction_Score': satisfaction_score,
        'Usage_Drop': usage_drop,
        'Contract_Type_Monthly': contract_monthly,
        'Contract_Type_Quaterly': contract_quaterly
    }])
    
   
    churn_prob = model.predict(input_df)[0]
    

    customer_threshold = 0.45
    prediction = 1 if churn_prob > customer_threshold else 0
    
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **Churned** (High Risk — Churn Probability: **{churn_prob*100:.1f}%** | Threshold: 45%)")
    else:
        st.success(f"✅ **Not Churned** (Safe Customer — Churn Probability: **{churn_prob*100:.1f}%** | Threshold: 45%)")

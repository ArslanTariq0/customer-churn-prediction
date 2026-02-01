import streamlit as st
import pandas as pd
import joblib

# Load saved model & tools
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("Customer Churn Prediction Demo")
st.write("Enter customer details to see churn risk (Idea #36)")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 10.0, 150.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 800.0)
    
    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    # Build input dictionary (match dataset columns)
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
        'PaperlessBilling_Yes': 1 if paperless_billing == "Yes" else 0,
        'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        # Add more if you want (for better accuracy) – model fills rest with 0
    }

    df_input = pd.DataFrame([input_data])
    
    # Reindex to match training columns
    df_input = df_input.reindex(columns=feature_names, fill_value=0)
    
    # Scale
    scaled_input = scaler.transform(df_input)
    
    # Predict
    prob = model.predict_proba(scaled_input)[0][1]  # Probability of churn (Yes)
    risk = round(prob * 100, 1)
    
    st.subheader("Result")
    st.metric("Churn Risk", f"{risk}%")
    
    if prob > 0.5:
        st.error("**High Risk** – This customer is likely to churn soon. Consider: discount, better support, or loyalty offer.")
    else:
        st.success("**Low Risk** – Customer likely to stay.")

st.info("This is a simple demo using Random Forest on Telco Churn dataset. Accuracy ~80%.")
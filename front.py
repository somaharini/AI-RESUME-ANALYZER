import streamlit as st
import pandas as pd
import pickle
import numpy as np


with open("model.pkl","rb") as f:
    model=pickle.load(f)

st.set_page_config(page_title="Churn Analytics Pro", layout="wide")   
st.title("Customer Churn Intelligence Dashboard")
st.markdown("Predict the likelihood of customer churn using our advanced machine learning model. " \
"This dashboard provides insights into customer behavior and helps businesses " \
"make informed decisions to retain valuable customers.")
#----Sidebar----
st.sidebar.header("Individual Customer Analysis")
tenure= st.sidebar.slider("Tenure (in months)",0,72,12)
monthly_charges= st.sidebar.number_input("Monthly Charges($)",0,200,50)
contract= st.sidebar.selectbox("Contract Type",["Month-to-month","One year","Two year"])
#simple mapping for individual customer analysis
input_data=np.array([[tenure,monthly_charges,0 if contract == "Month-to-month"else(1 if contract == "One year"else 2)]+[0]*16])

if st.sidebar.button("Predict  Churn"):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]
    if prediction[0] == 1:
        st.sidebar.error(f"High Risk: {prob:.2%}chance of churn")
    else:
        st.sidebar.success(f"Low Risk: {prob:.2%} chance of churn")


st.subheader("Bulk Prediction")
uploaded_file = st.file_uploader("Upload customer csv file for batch processing",type=["csv"])

if uploaded_file:
    data=pd.read_csv(uploaded_file)
    st.write("Results Preview")
    st.dataframe(data.head())
    st.download_button("Doenload Processed Results","Results.csv","text/.csv")
            
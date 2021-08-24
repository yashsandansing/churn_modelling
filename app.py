import pickle
import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np



model = load_model("customer_retention_1.h5")
scaler = joblib.load("ann_scaler.pkl")
ohencoder = joblib.load("onehot_col_transformer.pkl")


def prediction(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    if Gender == "Male":
        Gender = 1
    else:
        Gender = 0
    
    if HasCrCard == "Yes":
        HasCrCard = 1
    else:
        HasCrCard = 0
        
    if IsActiveMember == "Yes":
        IsActiveMember = 1
    else:
        IsActiveMember = 0
    
    preds=[[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]]
    result = round(model.predict(scaler.transform(np.array(ohencoder.transform(preds))))[0][0]*100,2)

    return result


def main():       
    st.set_page_config(page_title="Churn App")
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:black;padding:13px;border-radius:20px"> 
    <h1 style ="color:white;text-align:center;">Customer Retention App</h1> 
    </div> 
    <p>Enter below information to see if the probability of a customer leaving the bank</p>
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
       
    # following lines create boxes in which user can enter data required to make prediction 
    CreditScore = st.slider("Enter customer's Credit Score",min_value=300,max_value=850,step=1)
    Geography = st.selectbox('Select the country where the customer is located',("France", "Spain", "Germany"))
    Gender = st.selectbox('Enter customer\'s gender',("Male","Female"))
    Age = st.slider("Enter customer's age",min_value=18,max_value=100,step=1)
    Tenure = st.slider("How many years has the customer been with the bank?",min_value=1,max_value=50,step=1)
    Balance = st.number_input("Enter customer\'s account balance")
    NumOfProducts = st.number_input("How many bank products has the customer purchased?")
    HasCrCard = st.selectbox("Does the customer have a credit card?",("Yes","No"))
    IsActiveMember = st.selectbox("Is the customer an active member of the bank?",("Yes","No"))
    EstimatedSalary = st.number_input("Enter customer's annual salary")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        
        if result<50:
            churnstatus="STAYING IN"
        else:
            churnstatus="LEAVING"
        
        st.success('The customer will be {} the bank'.format(churnstatus))
        st.success('The probability of the customer leaving the bank is {} %'.format(result))

if __name__=='__main__': 
    main()
        


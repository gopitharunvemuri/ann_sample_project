import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

with open("labelencoder.pkl", "rb") as f:
    labelencoder = pickle.load(f)

with open("ohe.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

CreditScore = st.slider("Credit Score" ,0, 100)
geography = st.selectbox(label = "Geography",options=ohe.categories_[0])
gender = st.selectbox(label = "Gender",options=["Male", "Female"])
age = st.number_input("Age")
tenure = st.slider("Tenure",0, 10)
balance = st.number_input("Balance")
NumOfProducts = st.number_input("NumOfProducts")
HasCrCard = st.number_input("HasCrCard")
IsActiveMember = st.number_input("IsActiveMember")
EstimatedSalary = st.number_input("EstimatedSalary")

input_data = {
    'CreditScore': CreditScore,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure':tenure,
    'Balance': balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,
    'EstimatedSalary': EstimatedSalary
}


df = pd.DataFrame([input_data.values()], columns = input_data.keys())
ann_model = load_model("model.h5")
df["Gender"] = labelencoder.transform(df["Gender"])
df_geography = ohe.transform([[geography]])
df_geography = df_geography.toarray()
df_geography = pd.DataFrame(df_geography, columns = ohe.get_feature_names_out())
df = pd.concat([df.drop(["Geography"], axis = 1), df_geography], axis = 1)

scaled_X = scaler.transform(df)

value = ann_model.predict(scaled_X)
st.write("The obtained value is "+str(value))
if value > 0.5:
    st.write("The customer is going to churn")
else:
    st.write("The customer is not going to churn")
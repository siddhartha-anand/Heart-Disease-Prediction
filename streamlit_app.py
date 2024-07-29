import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

st.title('Heart Disease Prediction')
st.write('Enter the following details.')

age = st.number_input('age')
sex = st.text_input('sex')
if sex.lower() == 'male':
    sex = 1
else:
    sex = 0
cp = st.number_input('Constrictive Pericarditis')   
trestbps = st.number_input('The Resting Blood Pressure')
chol = st.number_input('cholestrol')
fbs = st.number_input('fasting Blood Sugar')
restecg = st.number_input('Resting ECG')
thalach = st.number_input('Maximum Heart Rate Achieved/Thalach')
exang = st.number_input('Exercise induced Angina')
oldpeak = st.number_input('Oldpeak')
slope = st.number_input('slope')
ca = st.number_input('Calcium')
thal = st.number_input('Thalassemia')

input_data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)



heart_data = pd.read_csv('heart_disease_data.csv')

X = heart_data.drop(columns='target', axis = 1)
Y = heart_data['target']

X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size=0.25,stratify=Y,random_state=2)

model = LogisticRegression()

model.fit(X_train,Y_train)

predcition = model.predict(input_data_reshaped)
if st.button('predict'):
    if predcition[0]==0:
        st.write('The Person does not have a Heart Disease')
    else:
        st.write('The person has Heart Disease.')    

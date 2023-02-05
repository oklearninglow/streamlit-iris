import streamlit as st
import pandas as pd
from sklearn import datasets
from joblib import load

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost
from xgboost import XGBClassifier

st.write("""
# Simple Prediction App
This app predicts the **MBTI** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Bmi = st.sidebar.slider('Body Mass Index', 14.06, , 42.18)
    Pain1 = st.sidebar.slider('Pain1', 2.0, 4.4, 3.4)
    Pain2 = st.sidebar.slider('Pain2', 1.0, 6.9, 1.3)
    Pain3 = st.sidebar.slider('Pain3', 0.1, 2.5, 0.2)
    Pain4 = st.sidebar.slider('Pain4', 0.1, 2.5, 0.2)
    data = {'Bmi': sepal_length,
            'Pain1': sepal_width,
            'Pain2': petal_length,
            'Pain3': petal_width}
            'Pain4': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

from joblib import load

# Load the saved StackingClassifier model from disk
loaded_stacking = load('best_model.joblib')

X = df.data
Y = df.target


loaded_stacking.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(data.target_names)

st.subheader('Prediction')
st.write(data.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
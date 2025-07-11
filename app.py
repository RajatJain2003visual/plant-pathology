import streamlit as st
import pickle
import functions
from functions import extract_features
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

with open("classifier.pkl","rb") as f:
    classifier = pickle.load(f)

st.title("Plant Disease classifier")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    with open("sample.jpg","wb") as f:
        f.write(uploaded_file.read())
    st.image("sample.jpg")
    labels = ['Healthy','Multiple_diseases','Rust','Scab']
    x = np.array([extract_features("sample.jpg")])
    prediction = labels[classifier.predict(x)[0]]
    st.success(f"Prediction : {prediction}")
    os.remove("sample.jpg")

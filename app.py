import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("plant_model.pkl","rb"))

st.title("Plant Disease Detection (KMeans)")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")

if st.button("Predict"):
    data = np.array([[f1, f2, f3]])
    pred = model.predict(data)
    st.success(f"Cluster: {pred[0]}")

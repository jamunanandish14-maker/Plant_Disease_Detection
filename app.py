import streamlit as st
import numpy as np
from sklearn.cluster import KMeans

st.title("Plant Disease Detection (KMeans)")

# recreate small demo-trained model
X = np.array([
    [5,3,1],
    [10,8,2],
    [50,60,55],
    [80,90,85],
    [20,25,22],
    [70,65,68]
])

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")

labels = {
    0: "Healthy",
    1: "Early Blight",
    2: "Late Blight"
}

if st.button("Predict"):
    data = np.array([[f1, f2, f3]])
    pred = model.predict(data)
    disease = labels[int(pred[0])]
    st.success(f"Disease: {disease}")

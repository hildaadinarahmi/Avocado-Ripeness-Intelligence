import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

# Check if model files exist
if os.path.exists("rf_model.pkl") and os.path.exists("scaler.pkl"):
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
else:
    st.error("❌ Model files not found. Please ensure 'rf_model.pkl' and 'scaler.pkl' are in the app directory.")
    st.stop()

# Page config
st.set_page_config(
    page_title="🥑 Avocado Ripeness Classifier",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Banner
st.markdown("""
    <h1 style='text-align: center; color: #228B22;'>🥑 Avocado Ripeness Classifier</h1>
    <p style='text-align: center;'>Predict the ripeness of avocados based on physical characteristics & sound</p>
""", unsafe_allow_html=True)

# Avocado Image
st.image("https://cdn.pixabay.com/photo/2016/03/05/19/02/avocado-1238257_1280.jpg", width=300, caption="Ready to ripen!", use_container_width=True)

# Sidebar input
st.sidebar.header("🧪 Avocado Feature Input")
with st.sidebar:
    st.image("https://i.ibb.co/NrXTVb4/avocado.png", use_container_width=True)
    st.markdown("## Avocado Info")
    st.markdown("💡 Classification of avocado ripeness levels.")
    selected_model = st.selectbox("🔍 Choose Model", ["Random Forest", "SVM", "KNN"])

firmness = st.sidebar.slider("💪 Firmness Score", 0.0, 10.0, 5.0)
hue = st.sidebar.slider("🌈 Hue", 0.0, 360.0, 120.0)
saturation = st.sidebar.slider("🎨 Saturation", 0.0, 1.0, 0.5)
brightness = st.sidebar.slider("🔆 Brightness", 0.0, 1.0, 0.5)
color_intensity = st.sidebar.slider("🟢 Color Intensity", 0.0, 100.0, 50.0)
sound_ratio = st.sidebar.slider("🔊 Sound Ratio", 0.0, 1.0, 0.5)
density = st.sidebar.slider("⚖️ Density", 0.0, 2.0, 1.0)

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    st.pyplot(fig)

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("📋 Preview Data:", data.head())

# Predict button
if st.sidebar.button("🔍 Predict Ripeness"):
    input_data = np.array([[firmness, hue, saturation, brightness, color_intensity, sound_ratio, density]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    label_dict = {0: "🟢 Hard", 1: "🟡 Ripe", 2: "🔴 Overripe"}
    st.success(f"🍃 Prediction Result: **{label_dict[prediction]}**")
    st.balloons()
else:
    st.info("Enter features in the sidebar, then click '🔍 Predict Ripeness' to start.")

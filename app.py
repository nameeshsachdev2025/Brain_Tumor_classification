import streamlit as st
import numpy as np
from PIL import Image
import time
from keras.models import load_model
from keras.utils import img_to_array

# App Configuration
st.set_page_config(page_title="🧠 Brain Tumor Classifier", layout="centered", page_icon="🧠")

# Title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Brain Tumor Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an MRI scan to predict the type of brain tumor using deep learning.</p>", unsafe_allow_html=True)

# Sidebar Medical Disclaimer
def show_disclaimer():
    with st.sidebar:
        st.markdown("### ⚠️ Medical Disclaimer")
        st.markdown("""
        This tool is for **research and educational purposes only**.  
        It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
        Always consult a qualified medical professional for health concerns.
        """)

show_disclaimer()

# Class Names
class_names = ['Glioma', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Load model (cached)
@st.cache_resource
def load_brain_model():
    try:
        model = load_model('brain_tumor_model.keras', compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_brain_model()
if model is None:
    st.stop()

# Image Preprocessing
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# File Uploader
st.markdown("### 📤 Upload an MRI Image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_display = Image.open(uploaded_file).convert('RGB')
    st.image(image_display, caption='🖼️ Uploaded MRI', use_column_width=True)

    with st.spinner('🔍 Analyzing the MRI scan...'):
        time.sleep(1)
        processed_image = preprocess_image(image_display)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    st.success("✅ Analysis Complete!")

    # Results Section
    st.markdown("---")
    st.markdown("### 🧾 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔬 Predicted Class")
        st.markdown(f"<div style='font-size: 24px; font-weight: bold; color: #4B8BBE;'>{predicted_class}</div>", unsafe_allow_html=True)
        st.metric("🔢 Confidence Score", f"{confidence:.2f}%")

        if predicted_class != 'No Tumor':
            st.warning("⚠️ Please consult a medical professional for further evaluation.")

    with col2:
        st.markdown("#### 📊 Probability Distribution")
        for class_name, prob in zip(class_names, predictions[0]):
            st.progress(int(prob * 100), text=f"{class_name}: {prob*100:.2f}%")

    st.info("""
    ℹ️ **Note:** This prediction is based on a machine learning model.  
    It is not intended for diagnostic use.  
    Always consult with a radiologist or neurosurgeon for clinical decisions.
    """)


import streamlit as st
import numpy as np
from PIL import Image
import time
from keras.models import load_model
from keras.utils import img_to_array

# Page Configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    :root {
        --primary: #6a5acd;
        --secondary: #9370db;
        --accent: #e6e6fa;
        --text: #333333;
        --light: #f8f9fa;
        --danger: #dc3545;
        --success: #28a745;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 2rem;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .prediction-card {
        border-left: 5px solid var(--primary);
    }
    
    .confidence-meter {
        height: 10px;
        background: linear-gradient(90deg, var(--danger), #ffc107, var(--success));
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .file-uploader {
        border: 2px dashed var(--secondary);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    .tumor-type {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin-right: 0.5rem;
        display: inline-block;
    }
    
    .glioma { background-color: #fff0f0; color: #d33; }
    .meningioma { background-color: #f0f8ff; color: #339; }
    .pituitary { background-color: #f0fff0; color: #393; }
    .no-tumor { background-color: #f5f5f5; color: #555; }
    
    .progress-container {
        margin-bottom: 1rem;
    }
    
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
    }
    
    .progress-bar {
        height: 8px;
        border-radius: 4px;
        background-color: #e9ecef;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--secondary), var(--primary));
    }
    
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header">
        <h1 style="margin:0; font-size: 2.5rem;">🧠 Brain Tumor Classifier</h1>
        <p style="margin:0; opacity: 0.9; font-size: 1.1rem;">
            Advanced MRI analysis using deep learning to detect and classify brain tumors
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with modern design
def show_disclaimer():
    with st.sidebar:
        # Medical Disclaimer
        st.markdown("""
            <div style="margin-bottom: 2rem;">
                <h3 style="color: #6a5acd;">Medical Disclaimer</h3>
                <div style="background-color: #fff8e1; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 0 8px 8px 0;">
                    <p style="margin-bottom: 0.5rem; color: #d32f2f;">
                        <strong>⚠️ This tool is for research purposes only.</strong>
                    </p>
                    <p style="font-size: 0.9rem; margin-bottom: 0; color: #333;">
                        Not a substitute for professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        
        st.markdown("""
            <div style="margin-top: 2rem;">
                <h4 style="color: var(--primary);">How It Works</h4>
                <ol style="font-size: 0.9rem; padding-left: 1.2rem;">
                    <li>Upload an MRI scan (JPEG/PNG)</li>
                    <li>Our AI analyzes the image</li>
                    <li>Get instant classification results</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

show_disclaimer()

# Class Labels with color coding
class_names = ['Glioma', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
class_colors = {
    'Glioma': 'glioma',
    'Meningioma Tumor': 'meningioma',
    'No Tumor': 'no-tumor',
    'Pituitary Tumor': 'pituitary'
}

# Load model with cache
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

# Preprocess uploaded image
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    # Simplified File Uploader with direct integration
    uploaded_file = st.file_uploader(
        "Upload MRI Scan (JPG, JPEG, PNG)", 
        type=["jpg", "jpeg", "png"],
        key="mri_uploader"
    )
    
    if uploaded_file:
        image_display = Image.open(uploaded_file).convert('RGB')
        st.image(image_display, caption='Uploaded MRI Image', use_container_width=True)

    else:
        st.markdown("""
            <div style="text-align: center; padding: 2rem; border: 2px dashed #ccc; border-radius: 8px;">
                <p style="color: #666;">No image uploaded yet</p>
            </div>
        """, unsafe_allow_html=True)

with col2:
    if uploaded_file:
        # Processing and Results Section
        with st.spinner('🔍 Analyzing MRI scan...'):
            time.sleep(1)  # Simulate processing time
            processed_image = preprocess_image(image_display)
            predictions = model.predict(processed_image)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
        
        st.markdown(f"""
            <div class="card prediction-card">
                <h3 style="margin-top: 0; color: var(--primary);">Analysis Results</h3>
                <div style="margin-bottom: 1.5rem;">
                    <span class="tumor-type {class_colors[predicted_class]}">{predicted_class}</span>
                    <div style="margin-top: 1rem;">
                        <div class="progress-label">
                            <span>Confidence</span>
                            <span>{confidence:.1f}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {confidence}%"></div>
                        </div>
                    </div>
                </div>
        """, unsafe_allow_html=True)
        
        if predicted_class != 'No Tumor':
            st.warning("**Medical Attention Recommended** - Please consult a neurologist for further evaluation.")
        else:
            st.success("**No tumor detected** - However, regular checkups are recommended.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Detailed Probabilities
        st.markdown("""
            <div class="card">
                <h4 style="margin-top: 0; color: var(--primary);">Detailed Probabilities</h4>
        """, unsafe_allow_html=True)
        
        for class_name, prob in zip(class_names, predictions[0]):
            percentage = prob * 100
            st.markdown(f"""
                <div class="progress-container">
                    <div class="progress-label">
                        <span>{class_name}</span>
                        <span>{percentage:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {percentage}%"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Placeholder when no image is uploaded
        st.markdown("""
            <div class="card" style="text-align: center; padding: 3rem 1rem; background-color: #f3f0ff; border-radius: 10px;">
                <img src="https://img.icons8.com/ios/100/6a5acd/brain-scan.png" width="80" style="opacity: 0.7;">
                <h4 style="color: #6a5acd; margin-top: 1rem;">No Image Uploaded</h4>
                <p style="color: #555; opacity: 0.7;">Upload an MRI scan to begin analysis</p>
            </div>
        """, unsafe_allow_html=True)




# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; color: #666; font-size: 0.9rem;">
        <p>This AI tool is designed to assist medical professionals, not replace them.</p>
        <p>For accurate diagnosis, always consult with a qualified healthcare provider.</p>
    </div>
""", unsafe_allow_html=True)



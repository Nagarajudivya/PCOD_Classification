import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf 

# Load your model
model = tf.keras.models.load_model('C:/Users/nagar/OneDrive/Desktop/Paa Project/app.py')  # Load from your .h5 file


# Page configuration
st.set_page_config(page_title="PCOS Diagnosis", layout="wide")
st.title("PCOS Diagnosis from Ultrasound Images")

# Image upload
uploaded_image = st.file_uploader("Upload Ultrasound Image", type=["jpg", "png", "jpeg"])

# Prediction
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize as per your model's requirement
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 128, 128, 1)  # Reshape for prediction

    # Make predictions
    y_pred = model.predict(image_array)
    probability = y_pred[0][0]

    # Display results
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if probability < 0.5:  
        st.success(f"PCOS detected with probability {100*(1- y_pred)}")
    else:
        st.info(f"No PCOS detected with probability {100 *( y_pred )}")

# Additional information (optional)
with st.expander("About PCOS"):
    st.write("Polycystic ovary syndrome (PCOS) is a hormonal disorder...")

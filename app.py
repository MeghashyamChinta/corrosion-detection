import streamlit as st
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# --- Load the Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"saved_model.h5")
    return model

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    image = image.resize((128, 128))
    image = img_to_array(image)
    image = image / 255.0  # Normalization
    image = np.expand_dims(image, axis=0)
    return image

# --- Streamlit UI ---
st.title("Vessel Corrosion Detection App ⚓")
st.markdown("Upload a vessel photo and detect **Corrosion vs No Corrosion** using a trained CNN model.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0][0]

            label = "NOCORROSION" if prediction > 0.5 else "CORROSION"
            confidence = prediction if prediction > 0.5 else 1 - prediction

            st.success(f"**Prediction: {label}**")
            st.write(f"Confidence: `{confidence:.2%}`")

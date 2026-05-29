import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = tf.keras.models.load_model("saved_models/brain_tumor_model.keras")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # simple confidence bar
    st.progress(float(confidence / 100))
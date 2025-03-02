import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("content/sign_language_cnn.h5")

# Define class labels
label_dict = {i: label for i, label in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                                                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
                                                  'T', 'U', 'V', 'W', 'X', 'Y'])}

def preprocess_image(image):
    img = image.resize((32, 32))  # Resize to match training size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

st.title("Hand Sign Language Recognition")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True) 
    
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_dict.get(predicted_class_index, "Unknown")

    st.write(f"Predicted Sign: {predicted_label}")

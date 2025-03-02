import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("content/sign_language_cnn.h5")

# Define class labels (A-Z excluding J and Z)
label_dict = {i: label for i, label in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                                                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
                                                  'T', 'U', 'V', 'W', 'X', 'Y'])}

def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the model's training format.
    Converts to grayscale, resizes to 28x28, normalizes pixel values.
    """
    img = image.convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match training size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (28, 28, 1)
    return img_array

st.title("🖐️ Hand Sign Language Recognition")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True) 
    
    img_array = preprocess_image(image)
    
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_dict.get(predicted_class_index, "Unknown")

    st.subheader(f"🔍 **Predicted Sign:** {predicted_label}")

    # Debugging: Show raw prediction probabilities
    st.write("📊 **Prediction Probabilities:**", prediction.tolist())

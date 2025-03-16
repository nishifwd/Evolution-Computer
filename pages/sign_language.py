import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("content/sign_language_cnn.h5")

# Define class labels (A-Z excluding J and Z)
label_dict = {i: label for i, label in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                                                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
                                                  'T', 'U', 'V', 'W', 'X', 'Y'])}

def preprocess_image(image):
    """Ensure identical preprocessing as in Colab."""
    if image.mode != "RGB":
        image = image.convert("RGB")  # Convert only if necessary
    img = image.resize((32, 32))  # Resize to match CNN input size
    img_array = image.img_to_array(img)  # Use Keras method
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app
st.title("🖐️ Hand Sign Language Recognition")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="📸 Uploaded Image", use_container_width=True) 
    
    # Preprocess and make prediction
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_label = label_dict.get(predicted_class_index, "Unknown")
    
    # Display results
    st.subheader(f"🔍 **Predicted Sign:** {predicted_label}")
else:
    st.warning("Please upload an image file.")

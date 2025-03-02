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
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_dict.get(predicted_class_index, "Unknown")
    
    # Display results
    st.subheader(f"🔍 **Predicted Sign:** {predicted_label}")

    # Show prediction probabilities as a bar chart
    st.subheader("📊 Prediction Confidence")

    # Convert label_dict keys to a sorted list of class labels
    sorted_labels = [label_dict[i] for i in range(len(label_dict))]
    
    # Plot prediction probabilities
    fig, ax = plt.subplots()
    ax.bar(sorted_labels, prediction[0])
    ax.set_xlabel("Class Labels")
    ax.set_ylabel("Confidence")
    plt.xticks(rotation=90)
    st.pyplot(fig)

else:
    st.warning("Please upload an image file.")

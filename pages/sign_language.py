import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model (after saving in the notebook)
model = tf.keras.models.load_model('hand_sign_language_model.h5')

# Streamlit app for user to upload an image
st.title('Hand Sign Language Recognition')

# Allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image before making predictions
    img_array = load_and_preprocess_image(uploaded_file)

    # Make predictions with the loaded model
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Get the class label (you should have your class labels from label_dict)
    class_labels = list(label_dict.keys())
    predicted_class_label = class_labels[predicted_class_index]

    st.write(f'Predicted Class: {predicted_class_label}')


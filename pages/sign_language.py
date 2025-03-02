import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model (after saving in the notebook)
model = tf.keras.models.load_model('content/hand_sign_language_model.h5')

# Define the image preprocessing function
def load_and_preprocess_image(image_path, target_size=(32, 32)):  # Resize to match your model's input size
    img = Image.open(image_path).resize(target_size)  # Open and resize image
    img_array = np.array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define the label_dict (mapping integer to class label)
label_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# Streamlit app for user to upload an image
st.title('Hand Sign Language Recognition')

# Allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image before making predictions
    img_array = load_and_preprocess_image(uploaded_file)

    # Make predictions with the loaded model
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Get the class label (you should have your class labels from label_dict)
    class_labels = list(label_dict.values())  # Get class labels from the dictionary
    predicted_class_label = class_labels[predicted_class_index]

    st.write(f'Predicted Class: {predicted_class_label}')

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("saved_models/brain_tumor_model.keras")

# Class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load image
img_path = "test.jpg"   # change to your image name

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
# Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Get result
predicted_class = class_names[np.argmax(prediction)]

""" print("Prediction:", predicted_class)
print("Confidence:", np.max(prediction) * 100) """

print("Prediction:", predicted_class)
print("Confidence:", round(float(np.max(prediction) * 100), 2), "%")
import os
import numpy as np

# noinspection PyUnresolvedReferences
from keras.models import load_model
# noinspection PyUnresolvedReferences
from keras.preprocessing import image

model_for_prediction = load_model(os.path.join('models','diagram_classification_model.keras'))
model_for_prediction.summary()


# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(200, 200))  # Assuming input size is 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Function to make predictions
def predict_image(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Make predictions
    predictions = model_for_prediction.predict(processed_image)

    return predictions


# Provide the path to your image
image_path = '{PATH_TO_IMAGE}'

# Make predictions on the image
predictions = predict_image(image_path)

# Print the predicted result
print(predictions)

prediction_percentages = predictions * 100
formatted_prediction = ["{:.2f}%".format(prob) for prob in prediction_percentages[0]]
labels = ['Line', 'Pie']
# Print the formatted prediction
for label, percentage in zip(labels, prediction_percentages[0]):
    print(f"{label}: {percentage:.2f}%")
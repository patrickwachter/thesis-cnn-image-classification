import os
import numpy as np

from io import BytesIO
# noinspection PyUnresolvedReferences
from keras.models import load_model
# noinspection PyUnresolvedReferences
from keras.preprocessing import image
from flask import Flask, request, jsonify

labels = ['Line', 'Pie']
model_for_prediction = load_model(os.path.join('models','diagram_classification_model.keras'))
app = Flask(__name__)

@app.route('/predictChart', methods=['POST'])
def predictChart():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # preprocess image
    img_bytes = file.read()
    img_stream = BytesIO(img_bytes)
    img = image.load_img(img_stream, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # model prediction
    predictions = model_for_prediction.predict(img_array)
    prediction_percentages = predictions * 100
    predictions_dict = {label: "{:.2f}".format(float(percentage)) for label, percentage in zip(labels, prediction_percentages[0])}

    # Find the class with the highest percentage
    prediction = labels[np.argmax(predictions)]

    return jsonify({'prediction': prediction, 'filename': file.filename, 'predictions': predictions_dict})

if __name__ == '__main__':
    app.run(debug=False)

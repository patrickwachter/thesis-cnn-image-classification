# thesis-cnn-image-classification

This is a simple Flask web application that classifies diagrams into two categories: **Line** and **Pie**. The app utilizes a pre-trained Keras model for image classification and provides an interface for users to upload images and receive predictions.

1. **App with GUI**: A web-based interface for user interaction.
2. **API-only App**: A command-line interface for making predictions via HTTP requests, suitable for cURL or other API clients.

## Features

- **Web Interface**: An HTML-based interface for image upload and classification.
- **API Interface**: A RESTful API endpoint for programmatically submitting image files and receiving predictions.
- **Pre-trained Model**: Both applications use the same pre-trained Keras model to classify diagrams.
- **Prediction**: Returns the predicted class along with confidence percentages for both classes.

## Prerequisites

Make sure you have the following installed:
- Python 3.x
- Flask
- Keras
- NumPy
- TensorFlow (Keras backend)

## Installation

1. Clone the repository to your local machine.

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required packages using pip:

   ```bash
   pip install Flask keras numpy tensorflow
   ```

3. Ensure that you have the trained model file named `diagram_classification_model.keras` located in the `models` directory.

## Application 1: Web Interface

This Flask app provides an HTML interface for users to upload diagram images and receive predictions.

### Usage

1. Start the application:

   ```bash
   python model-predict-endpoint-gui.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/` or `http://localhost:5000/` to access the application.

3. Upload an image of a diagram (either Line or Pie) using the provided form. Although all images types are accepted, the application was tested mostly with png. ![Alt text](form.png?raw=true "Form")

5. Submit the form to get the prediction, which will return the predicted class and confidence percentages for both classes. 

### Example of a Prediction Response

![Alt text](prediction.png?raw=true "Prediction")

## Application 2: API-only Interface

This version of the app provides a `/predictChart` endpoint that can be called with HTTP POST requests to receive predictions. This app does not provide a GUI, making it suitable for use with cURL, Postman, or other API clients.

### Usage

1. Start the application:

   ```bash
   python model-predict-endpoint.py
   ```

2. Use cURL or any HTTP client to send a POST request with an image file to the `/predictChart` endpoint. Although all images types are accepted, the application was tested mostly with png.

   #### Example cURL Command

   ```bash
   curl -X POST http://127.0.0.1:5000/predictChart -F "image=@/path/to/your/image.png"
   ```

3. The response will be a JSON object containing the predicted class and confidence percentages.

### Example of a Prediction Response

```json
{
  "prediction": "Pie",
  "filename": "example_diagram.png",
  "predictions": {
    "Line": "40.12",
    "Pie": "59.88"
  }
}
```

## Code Explanation

- **Model Loading**: Both apps load a pre-trained Keras model from the `models` directory to classify uploaded images.
- **Image Preprocessing**: The uploaded image is resized and converted to an array suitable for input into the model.
- **Prediction**: The model generates predictions for each class (Line and Pie), and the app returns the class with the highest percentage.

- **Imports**:
  - `Flask`: The web framework used to create the web application.
  - `request`: To handle incoming requests and uploaded files.
  - `jsonify`: To convert Python dictionaries to JSON format for responses.
  - `render_template`: To render HTML templates.
  - `os`: For file path management.
  - `numpy`: For numerical operations and array manipulations.
  - `BytesIO`: To handle image bytes for processing.
  - `load_model` and `image`: Functions from Keras for loading the model and preprocessing images.

- **Model Loading**:
  The Keras model is loaded from the `models` directory at the start of the application.

- **Routes**:
  - **Index Route (`/`)**: Renders the main HTML page.
  - **Prediction Route (`/predictChart`)**: Handles the image upload and prediction.

# thesis-cnn-image-classification

This is a simple Flask web application that classifies diagrams into two categories: **Line** and **Pie**. The app utilizes a pre-trained Keras model for image classification and provides an interface for users to upload images and receive predictions.

## Features

- User-friendly HTML interface for image upload.
- Image classification using a pre-trained model.
- Returns the predicted class along with confidence percentages for both classes.

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- Flask
- Keras
- NumPy
- TensorFlow (Keras backend)

## Installation

1. Clone the repository or download the code files to your local machine.

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required packages using pip:

   ```bash
   pip install Flask keras numpy tensorflow
   ```

3. Ensure that you have the trained model file named `diagram_classification_model.keras` located in the `models` directory.

## Usage

1. Start the Flask application:

   ```bash
   python flask_app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/` or `http://localhost:5000/` (this URL is printing in the server console on startup) to access the application.

3. Upload an image of a diagram (either Line or Pie, all types of image files are accepted but only .png was tested in depth) using the provided form.

4. Submit the form to get the prediction. The app will return the predicted class along with confidence percentages.

## Code Explanation

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
    - Checks if an image file is present.
    - Reads and preprocesses the image.
    - Uses the loaded model to predict the class of the image.
    - Returns the prediction results in JSON format.

## Example of a Prediction Response

```json
{
  "prediction": "Line",
  "filename": "uploaded_diagram.png",
  "predictions": {
    "Line": "75.23",
    "Pie": "24.77"
  }
}
```

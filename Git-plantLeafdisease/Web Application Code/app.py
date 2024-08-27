from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Initialize Flask application
app = Flask(__name__)

# Load the trained model for disease detection
model = tf.keras.models.load_model("model.h5")  # Load the model from model.h5

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict(image_path):
    try:
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array)
        # Assuming your model outputs probabilities for each class
        # Modify this according to your model's output
        class_labels = ['Healthy', 'Powdery', 'Rust']
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)
        return predicted_class, confidence
    except Exception as e:
        print("Error:", e)
        return None, None

# Function to detect leaf in the image
def detect_leaf(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            return True
        else:
            return False
    except Exception as e:
        print("Error:", e)
        return False

# Route to upload page
@app.route('/')
def upload_form():
    return render_template('upload.html', uploaded_image=None)

# Route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('upload.html', error='No file part', uploaded_image=None)

    file = request.files['file']

    if file.filename == '':
        return render_template('upload.html', error='No selected file', uploaded_image=None)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)
        
        if detect_leaf(filepath):
            predicted_disease, confidence = predict(filepath)
            if predicted_disease is not None:
                return redirect(url_for('result', image=filename, disease=predicted_disease, confidence=confidence))
            else:
                return redirect(url_for('no_disease_detected', image=filename))
        else:
            return redirect(url_for('no_leaf_detected', image=filename))
    else:
        return render_template('upload.html', error='File type not allowed', uploaded_image=None)

# Route to display result when no leaf is detected
@app.route('/no_leaf_detected')
def no_leaf_detected():
    image_url = url_for('static', filename='uploads/' + request.args['image'])
    return render_template('no_leaf_detected.html', image_url=image_url)

# Route to display result when no disease is detected
@app.route('/no_disease_detected')
def no_disease_detected():
    image_url = url_for('static', filename='uploads/' + request.args['image'])
    return render_template('no_disease_detected.html', image_url=image_url)

# Route to display result
@app.route('/result')
def result():
    image_url = url_for('static', filename='uploads/' + request.args['image'])
    predicted_disease = request.args['disease']
    confidence = request.args['confidence']
    return render_template('result.html', image_url=image_url, predicted_disease=predicted_disease, confidence=confidence)

# Route to upload another file
@app.route('/upload-another')
def upload_another():
    return redirect(url_for('upload_form'))

if __name__ == "__main__":
    app.run(debug=True)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import base64

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained model
model = tf.keras.models.load_model('seismic_waveform_magnitude_predictor.h5')

# Load the dataset
dataset_path = 'magnitudes.csv'
dataset = pd.read_csv(dataset_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_and_preprocess_image(file):
    img = load_img(BytesIO(file.read()), target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array

def get_actual_magnitude(trace_name):
    trace_name = trace_name.rsplit('.', 1)[0]
    row = dataset[dataset['trace_name'].str.contains(trace_name, na=False)]
    if not row.empty:
        return row.iloc[0]['source_magnitude']
    return None

def generate_pdf(predicted_magnitude, actual_magnitude, img_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    title = "Prediction Report"
    c.setFont("Times-Bold", 20)
    title_width = c.stringWidth(title, "Times-Bold", 20)
    c.drawString((width - title_width) / 2, height - 50, title)
    
    # Prediction Details
    c.setFont("Times-Bold", 14)
    c.drawString(30, height - 100, f"Predicted Magnitude: {predicted_magnitude}")
    c.drawString(30, height - 120, f"Actual Magnitude: {actual_magnitude}")
    c.drawString(30, height - 140, "Akurasi: 91.60%")
    
    # Image
    image = ImageReader(BytesIO(img_data))
    c.drawImage(image, 30, height - 700, width=550, height=800, preserveAspectRatio=True)
    
    c.showPage()
    c.save()
    buffer.seek(0)
    
    return buffer

@app.route('/download/<predicted_magnitude>/<actual_magnitude>', methods=['POST'])
def download_pdf(predicted_magnitude, actual_magnitude):
    img_data = request.form['img_data'].encode('utf-8')
    img_data = base64.b64decode(img_data)
    buffer = generate_pdf(predicted_magnitude, actual_magnitude, img_data)
    return send_file(buffer, as_attachment=True, download_name=f'prediction_report.pdf', mimetype='application/pdf')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            img_data = file.read()
            file.seek(0)  # Reset the file pointer to the beginning of the file

            img_array = load_and_preprocess_image(file)
            img_array = np.expand_dims(img_array, axis=0)

            y_pred = model.predict(img_array)
            predicted_magnitude = y_pred[0][0]

            actual_magnitude = get_actual_magnitude(file.filename)
            if actual_magnitude is None:
                actual_magnitude = "Unknown"

            img_base64 = base64.b64encode(img_data).decode('utf-8')

            return render_template('index.html', prediction=predicted_magnitude, actual_magnitude=actual_magnitude, img_data=img_base64)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

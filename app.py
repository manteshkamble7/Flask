from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)
model = None

# Load the .h5 model file
def load_deep_learning_model():
    global model
    model = load_model('model977.h5')
    print("Model loaded successfully!")

# Function to process an image and make a prediction
def predict_image(image_path):
    try:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize the image
        prediction = model.predict(image)[0]
        return prediction
    except Exception as e:
        print("Error predicting the image:", str(e))
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        if file:
            image_path = os.path.join('uploads', file.filename)
            file.save(image_path)
            prediction = predict_image(image_path)
            os.remove(image_path)  # Remove the uploaded file after prediction
            
            # Customize the output format according to your model's task (e.g., classification, regression, etc.)
            result = f"Prediction: {prediction}"
            return result

    return render_template('index.html')

if __name__ == '__main__':
    load_deep_learning_model()
    app.run(debug=True)

from flask import Flask, request, jsonify
from tensorflow.keras.applications.mobilenet import preprocess_input
from flask_restful import Api, Resource
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from flask_cors import CORS
import traceback

# Load the single combined model
model_path = 'model.h5'
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

img_width, img_height = 256, 256
classes = ['0Normal', '2Mild', '3Moderate', '4Severe']

app = Flask(__name__)
# Allow only the React app origin
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

api = Api(app)

class ImageProcessing(Resource):
    def __init__(self):
        super(ImageProcessing, self).__init__()
        self.model = model

    def post(self):
        try:
            print("Received POST request")
            if 'image' not in request.files:
                print("No image part in the request")
                return jsonify({'error': 'No image part in the request'}), 400

            image_file = request.files['image']
            if image_file.filename == '':
                print("No selected file")
                return jsonify({'error': 'No selected file'}), 400

            img = Image.open(image_file)
            img = img.resize((img_width, img_height))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            if self.model is None:
                print("Model not loaded correctly")
                return jsonify({'error': 'Model not loaded correctly'}), 500

            prediction = self.model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_disease = classes[predicted_class]
            
            print(f"Prediction: {predicted_disease}")
            response = jsonify({'prediction': predicted_disease})
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
            return response

        except Exception as e:
            traceback.print_exc()
            response = jsonify({'error': str(e)})
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
            return response, 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response

api.add_resource(ImageProcessing, '/process-image')

if __name__ == '__main__':
    app.run(debug=True)

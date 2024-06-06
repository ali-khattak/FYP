from flask import Flask, request, jsonify, render_template, redirect, url_for, flash,session
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from d import CORS
import traceback
import os
from flask_pymongo import PyMongo



model_path = 'model.h5'
print(model_path)
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

img_width, img_height = 256, 256

classes = ['0Normal', '2Mild', '3Moderate', '4Severe']

app = Flask(__name__)
app.secret_key = 'orthofyp'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

app.config['MONGO_URI'] = 'mongodb://localhost:27017/ortho'  
mongo = PyMongo(app)

try:
    # The ismaster command is cheap and does not require auth.
    mongo.db.command('ismaster')
    print('MongoDB connection successful')
except ConnectionFailure:
    print('MongoDB connection failed')

# Ensure CORS is properly configured
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    # MongoDB login logic
    users_collection = mongo.db.users
    user = users_collection.find_one({'username': username, 'password': password})
    
    if user:
        session['username'] = username  # Store username in session
        flash('Login successful!')
        return redirect(url_for('app_page'))
    else:
        flash('Invalid credentials. Try again.')
        return redirect(url_for('index'))

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']
    age = request.form['age']
    
    if username and password and age:
        # Insert the new user into MongoDB
        users_collection = mongo.db.users
        user_data = {
            'username': username,
            'password': password,  # Note: In a real app, never store plaintext passwords
            'age': age
        }
        users_collection.insert_one(user_data)
        flash('Signup successful! Redirecting to login.')
        return redirect(url_for('index'))
    else:
        flash('Please provide username, password, and age.')
        return redirect(url_for('index'))

@app.route('/app')
def app_page():
    return render_template('app.html')


@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        if 'username' not in session:
            return jsonify({'error': 'User not logged in'}), 401

        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        img = Image.open(image_file)
        img = img.resize((img_width, img_height))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        img.save(os.path.join('static', 'uploaded_image.jpg'))

        if model is None:
            return jsonify({'error': 'Model not loaded correctly'}), 500

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_disease = classes[predicted_class]

        # Store the prediction result in MongoDB
        predictions_collection = mongo.db.predictions
        prediction_data = {
            'username': session['username'],  # Include the username
            'filename': image_file.filename,
            'prediction': predicted_disease
        }
        predictions_collection.insert_one(prediction_data)

        return render_template('predicted.html', imageUrl=url_for('static', filename='uploaded_image.jpg'), prediction=predicted_disease)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response

if __name__ == '__main__':
    app.run(debug=True)
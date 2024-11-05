from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import shutil
import joblib

app = Flask(__name__, template_folder=r'C:\Users\Harshavelu\OneDrive\Desktop\template')
app.secret_key = 'c08698debcfc31310aa90c00c7df3f99'

UPLOAD_FOLDER = os.path.join(app.root_path, 'static')
TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, 'temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

try:
    base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
except Exception as e:
    print("Error loading VGG16 model:", e)
    exit()

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

users = {
    'author1': 'password1',  # You can add more authors and their passwords here
    'author2': 'password2'
}

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_malaria(image_array, base_model, ada_clf):
    features = base_model.predict(image_array).reshape(1, -1)
    ada_prediction = ada_clf.predict(features)
    return ada_prediction[0]

@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            flash('You were successfully logged in', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You were successfully logged out', 'success')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        flash('You need to log in first', 'error')
        return redirect(url_for('login'))

    if 'image' not in request.files:
        flash('No image uploaded', 'error')
        return redirect(url_for('index'))

    image_file = request.files['image']
    if image_file.filename == '':
        flash('No selected image', 'error')
        return redirect(url_for('index'))

    temp_file = os.path.join(TEMP_FOLDER, image_file.filename)
    image_file.save(temp_file)

    image_array = load_and_preprocess_image(temp_file)

    try:
        ada_clf_vgg16 = joblib.load(r'C:\Users\Harshavelu\Downloads\ada_clf_vgg16.joblib')
    except Exception as e:
        print("Error loading AdaBoost classifier:", e)
        flash('Error loading model', 'error')
        return redirect(url_for('index'))

    prediction = predict_malaria(image_array, base_model_vgg16, ada_clf_vgg16)

    folder = 'Parasite' if prediction == 1 else 'Uninfected'
    target_folder = os.path.join(UPLOAD_FOLDER, folder)
    os.makedirs(target_folder, exist_ok=True)
    target_path = os.path.join(target_folder, image_file.filename)
    shutil.move(temp_file, target_path)

    result = "Malaria Parasite detected." if prediction == 1 else "No Malaria Parasite detected."
    return render_template('result.html', result=result, image_path=target_path)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/author')
def author():
    # Add logic here to handle the author functionality
    return render_template('author.html')  # Render the HTML template for adding authors

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check if the username is already taken
        if username in users:
            flash('Username already taken', 'error')
            return redirect(url_for('signup'))
        # Store the new user's credentials
        users[username] = password
        flash('Account created successfully', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

if __name__ == '__main__':
    app.run(debug=True)
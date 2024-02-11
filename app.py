#Importing necessary Libraries
import tensorflow as tf
import keras
from flask import Flask , request, jsonify
from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array,get_file
from tensorflow.keras.models import load_model
import pickle
import io
import numpy as np
from arcface import arcface_loss
from flask_cors import CORS

#creating the flask app
app = Flask(__name__)
CORS(app)

# Loading the model
with tf.keras.utils.custom_object_scope({'arcface_loss': arcface_loss}):
    model = load_model('predictor.h5')

# Getting the dictioanry files of diseases names
with open('disease_names.pkl', 'rb') as pkl_file:
    disease_names = pickle.load(pkl_file)

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        file_contents = file.read()
        img = load_img(io.BytesIO(file_contents), target_size=(256, 256,3))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array.reshape(1, 256, 256, 3))
        pred = np.argmax(model.predict(img_array))
        disease_name = disease_names[pred]
        return jsonify({'prediction': disease_name})
        

if __name__ == '__main__':
    app.run(debug=True)

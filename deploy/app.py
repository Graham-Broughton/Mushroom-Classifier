# some utilities
import os, yaml
import numpy as np
import pickle
from .src.util import base64_to_pil


# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

#tensorflow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

config = yaml.safe_load(open("deploy/src/inat_config.YAML", 'rb'))
# Declare a flask app
app = Flask(__name__)

def get_image_class_dict():
    image_class_dict = pickle.load(open(config['LABEL_DICT_PATH'], 'rb'))
    return image_class_dict

def get_ImageClassifierModel():
    model = load_model(config['MODEL_PATH'])

    return model  

def model_predict(img, model):
    '''
    Prediction Function for model.
    Arguments: 
        img: is address to image
        model : image classification model
    '''
    img = img.resize((512, 512))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        
        # initialize model
        model = get_ImageClassifierModel()

        # Make prediction
        preds = model_predict(img, model)
        
        label_dict = get_image_class_dict()
        
        sorted_preds = preds[0].argsort()[::-1][:100]
        predictions = list(str(label_dict[pred]) for pred in sorted_preds[:3])
        probas = list(str(round(preds[0][pred] * 100, 4))+'%\n' for pred in sorted_preds[:3])

        preds = list(zip(predictions, probas))
        
        # Serialize the result, you can add additional fields
        return jsonify(result=preds)
    return None

if __name__ == '__main__':
    app.run(host="0.0.0.0")

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image
import uuid, os, yaml, pickle, json
from flask import Flask, request, jsonify, render_template

from form import ImageForm

config = yaml.safe_load(open('config.YAML'))
label_dict = pickle.load(open(config['LABEL_DICT_PATH'], 'rb'))

app = Flask(__name__)
UPLOAD_FOLDER = 'static/'

model = keras.models.load_model(config['MODEL_PATH'])

@app.route('/', methods=['GET', 'POST'])
def classify(file):
    form = ImageForm()
    if request.method == 'POST':
        img_file = form.image.data
        image_uuid = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, image_uuid)
        img_file.save(file_path)
        
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [512, 512])
        img = tf.expand_dims(img, 0)

        preds = model.predict(img)
        sorted_preds = preds[0].argsort()[::-1][:100]

        predictions = jsonify(dict({label_dict[pred]: round(preds[0][pred] * 100, 4) for pred in sorted_preds[:10]}))

        return predictions
    else:
        render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)
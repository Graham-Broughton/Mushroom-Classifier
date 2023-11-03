from os import environ
from pickle import load

import numpy as np
import requests
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, request
from PIL import Image
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

app = Flask(__name__)

# Load environment variables for Twilio
TWILIO_ACCOUNT_SID = environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = environ.get("TWILIO_AUTH_TOKEN")
# TWILIO_PHONE_NUMBER = environ.get("TWILIO_PHONE_NUMBER")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

class_dict = load(open("class_dict.pkl", "rb"))

# Load the model
model = tf.keras.models.load_model("model")


def topk(array, k, axis=-1, sorted=True):
    # Use np.argpartition is faster than np.argsort, but do not return the values in order
    # We use array.take because you can specify the axis
    partitioned_ind = np.argpartition(array, -k, axis=axis).take(
        indices=range(-k, 0), axis=axis
    )
    # We use the newly selected indices to find the score of the top-k values
    partitioned_scores = np.take_along_axis(array, partitioned_ind, axis=axis)

    if sorted:
        # Since our top-k indices are not correctly ordered, we can sort them with argsort
        # only if sorted=True (otherwise we keep it in an arbitrary order)
        sorted_trunc_ind = np.flip(np.argsort(partitioned_scores, axis=axis), axis=axis)

        # We again use np.take_along_axis as we have an array of indices that we use to
        # decide which values to select
        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
    else:
        ind = partitioned_ind
        scores = partitioned_scores

    return {"scores": scores, "indices": ind}


def local_model_prediction(image_url):
    # Download and preprocess the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((224, 224))  # Assume this is the size your model expects
    image_array = np.asarray(image) / 1.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    preds = model.predict(image_array)
    predictions = topk(preds[0], 3)

    if predictions["scores"][0] >= 0.90:
        predicted_class = predictions["indices"][0]
        message = (
            f"Your fun guy is {class_dict[predicted_class]} with >= 90% confidence!"
        )
    elif predictions["scores"][0] >= 0.60:
        predicted_class = predictions["indices"][0]
        message = f"Your fun guy probably {class_dict[predicted_class]} with 90% >= 60% confidence!\n2nd choice: {class_dict[predictions['indices'][1]]} with {predictions['scores'][1]*100:.2f}% confidence."
    elif predictions["scores"][0] >= 0.30:
        message = f"Im not too sure about this one, it might be {class_dict[predictions['indices'][0]]} with 60% >= 30% confidence!\n2nd choice: {class_dict[predictions['indices'][1]]} with {predictions['scores'][1]*100:.2f}% confidence."
    else:
        cls = predictions["indices"]
        pred = predictions["scores"]
        message = (
            "Sorry, I can't tell what this is.\n"
            + f"{class_dict[cls[0]]} with {pred[0]*100:.2f}% confidence.\n{class_dict[cls[1]]} with {pred[1]*100:.2f}% confidence.\n{class_dict[cls[2]]} with {pred[2]*100:.2f}% confidence."
        )

    # Map the predicted class to its label (modify this as per your labels)
    return message


@app.route("/sms", methods=["POST"])
def sms_response():
    response = MessagingResponse()
    response.message(
        "Please wait while we ID your fun guy... For best results, a side profile of the whole mushroom close up usually works, if not try a top or gill view."
    )

    # Extract the image URL from the incoming MMS
    if request.form["NumMedia"] != "0":
        image_url = request.form.get("MediaUrl0")

    try:
        # Get prediction from the local model
        prediction = local_model_prediction(image_url)

        # Send an SMS with the prediction
        # Respond to the text message.
        response.message(prediction)
        # response.message("Please wait while we ID your fun guy...")

    except Exception as error:
        print(f"Error: {error}")
        response.message("Sorry, something went wrong. Please try again.")

    return str(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

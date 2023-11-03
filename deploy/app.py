# from os import environ
from pickle import load

import numpy as np
import requests
import tensorflow as tf

# from dotenv import load_dotenv
from flask import Flask, request
from PIL import Image

# from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Load the model and class dictionary
class_d = load(open("class_dict.pkl", "rb"))
model = tf.keras.models.load_model("model")

# load_dotenv()
# Load environment variables for Twilio
# TWILIO_ACCOUNT_SID = environ.get("TWILIO_ACCOUNT_SID")
# TWILIO_AUTH_TOKEN = environ.get("TWILIO_AUTH_TOKEN")
# TWILIO_PHONE_NUMBER = environ.get("TWILIO_PHONE_NUMBER")
# twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def topk(array, k, axis=-1, sorted=True):
    """Returns the top-k scores and indices from the input array along the specified axis.

    Args:
        array (numpy.ndarray): The input array.
        k (int): The number of top scores to return.
        axis (int, optional): The axis along which to compute the top-k scores. Defaults to -1.
        sorted (bool, optional): Whether to return the scores in sorted order. Defaults to True.

    Returns:
        A dictionary containing the top-k scores and indices.
            - 'scores': A numpy.ndarray containing the top-k scores.
            - 'indices': A numpy.ndarray containing the indices of the top-k scores.
    """
    # np.argpartition is faster than np.argsort, but does not return the values in order
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


def model_predictions(image_url):
    """Predicts the top 3 most likely mushroom species from an image URL using a pre-trained model.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        list: A list of tuples, where each tuple contains a mushroom species name and its corresponding probability score.
    """
    # Download and preprocess the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((224, 224))  # Assume this is the size your model expects
    image_array = np.asarray(image) / 1.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    preds = model.predict(image_array)
    predictions = topk(preds[0], 3)
    return predictions


def evaluate_preds(preds, upper_lim=0.90, middle_lim=0.60, lower_lim=0.30):
    """Evaluate the predictions made by a classifier and return a message describing the predicted class.

    Args:
        preds (dict): A dictionary containing the predicted class indices and their corresponding scores.
        upper_lim (float, optional): The confidence threshold for a high-confidence prediction. Defaults to 0.90.
        middle_lim (float, optional): The confidence threshold for a medium-confidence prediction. Defaults to 0.60.
        lower_lim (float, optional): The confidence threshold for a low-confidence prediction. Defaults to 0.30.

    Returns:
        str: A message describing the predicted class and its confidence level.
    """
    if preds["scores"][0] >= upper_lim:
        predicted_class = preds["indices"][0]
        message = f"Your fun guy is {class_d[predicted_class]} with >= {int(upper_lim*100)}% confidence!"
    elif preds["scores"][0] >= middle_lim:
        predicted_class = preds["indices"][0]
        message = (
            f"Your fun guy is probably {class_d[predicted_class]} with {int(upper_lim*100)}% >= {int(middle_lim*100)}% confidence!\n"
            + f"2nd choice: {class_d[preds['indices'][1]]} with {preds['scores'][1]*100:.2f}% confidence."
        )
    elif preds["scores"][0] >= lower_lim:
        message = (
            f"Im not too sure about this one, it might be {class_d[preds['indices'][0]]} with {int(middle_lim*100)}% >= "
            + f"{int(lower_lim*100)}% confidence.\nMy 2nd choice: {class_d[preds['indices'][1]]} with {preds['scores'][1]*100:.2f}% confidence."
        )
    else:
        cls = preds["indices"]
        pred = preds["scores"]
        message = (
            "Sorry, I can't tell what this is.\n",
            [
                f"{class_d[cls[i]]} with {pred[i]*100:.2f}% confidence.\n"
                for i in range(3)
            ],
            # + f"{class_d[cls[0]]} with {pred[0]*100:.2f}% confidence.\n{class_d[cls[1]]} with {pred[1]*100:.2f}% confidence.\n{class_d[cls[2]]} with {pred[2]*100:.2f}% confidence."
        )
        message = message[0] + "".join(message[1])

    # Map the predicted class to its label (modify this as per your labels)
    return message


@app.route("/sms", methods=["POST"])
def sms_response():
    """Responds to an incoming SMS with a message requesting an image of a mushroom to be identified.

    Returns:
        str: A string representation of the response message to be sent back to the user.
    """
    response = MessagingResponse()
    response.message(
        "Please wait while we ID your fun guy... "
        + "For best results, a side profile of the whole mushroom close up usually works, if not try a top or gill view."
    )

    # Extract the image URL from the incoming MMS
    if request.form["NumMedia"] != "0":
        image_url = request.form.get("MediaUrl0")

    try:
        # Get prediction from the local model
        predictions = model_predictions(image_url)
        msg = evaluate_preds(predictions)

        # Send an SMS with the prediction
        # Respond to the text message.
        response.message(msg)
        # response.message("Please wait while we ID your fun guy...")

    except Exception as error:
        print(f"Error: {error}")
        response.message("Sorry, something went wrong. Please try again.")

    return str(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

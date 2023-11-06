import sys
from pickle import load

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, request
from PIL import Image
from twilio.twiml.messaging_response import MessagingResponse

sys.path.append("../training")
from config import GCFG

app = Flask(__name__)

# Load the model, class dictionary and config module
class_d = load(open("class_dict.pkl", "rb"))
model = tf.keras.models.load_model("model")
CFG = GCFG()


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


def model_predictions(image_url_list):
    """Predicts the top 3 most likely mushroom species from an image URL using a pre-trained model.

    Args:
        image_url_list (list): The URL of the image to be classified.

    Returns:
        list: A list of tuples, where each tuple contains a mushroom species name and its corresponding probability score.
    """
    # Download and preprocess the images
    imgs = list(
        map(
            lambda x: Image.open(requests.get(x, stream=True).raw).resize(
                CFG.IMAGE_SIZE
            ),
            image_url_list,
        )
    )
    imgs = np.array([np.array(img) for img in imgs])
    imgs = tf.data.Dataset.from_tensor_slices(imgs)
    imgs = imgs.map(lambda x: tf.cast(x, tf.float32))
    imgs = imgs.map(lambda x: tf.expand_dims(x, axis=0))

    # Predict
    preds = model.predict(imgs)
    predictions = topk(preds, 3)
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
    for i in range(preds["scores"].shape[0]):
        scores = preds["scores"][i]
        indx = preds["indices"][i]

        first, second, third = (
            (scores[0], indx[0]),
            (scores[1], indx[1]),
            (scores[2], indx[2]),
        )
        if first[0] >= upper_lim:
            message = f"Your fun guy is {class_d[first[1]]} with {int(first[0]*100)}% confidence!"
        elif first[0] >= middle_lim:
            message = (
                f"Your fun guy is probably {class_d[first[1]]} with {int(first[0]*100)}% confidence!\n"
                + f"2nd choice: {class_d[second[1]]} with {second[0]*100:.2f}% confidence."
            )
        elif first[0] >= lower_lim:
            message = (
                f"Im not too sure about this one, it might be {class_d[first[1]]} with {int(first[0]*100)}% confidence, \n"
                + f"or it could be {class_d[second[1]]} with {second[0]*100:.2f}% confidence."
            )
        else:
            message = (
                "Sorry, I can't tell what this is.\n",
                [
                    f"{class_d[rank[1]]} with {rank[0]*100:.2f}% confidence."
                    for rank in [first, second, third]
                ],
            )
            message = message[0] + "\n".join(message[1])

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
    num_photos = request.form["NumMedia"]
    img_urls = []
    if num_photos != "0":
        for idx in range(int(num_photos)):
            image_url = request.form.get(f"MediaUrl{idx}")
            img_urls.append(image_url)

    try:
        # Get prediction from the local model
        predictions = model_predictions(img_urls)
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

import json
from os import environ
from pickle import load

import numpy as np
import src.preprocessing as preprocessing
import uvicorn
from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()

root = environ.get("PYTHONPATH", ".").split(":")[0]
class_d = load(open(f"{root}/class_dict.pkl", "rb"))
model = preprocessing.get_model(f"{root}/model/")

IMAGE_SIZE = environ.get("IMAGE_SIZE", "[224, 224]")
IMAGE_SIZE = json.loads(IMAGE_SIZE)


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
        sorted_trunc_ind = np.flip(np.argsort(partitioned_scores, axis=axis), axis=axis)
        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
    else:
        ind = partitioned_ind
        scores = partitioned_scores

    return {"scores": scores, "indices": ind}


def model_predictions(dataset):
    """Predicts the top 3 most likely mushroom species from an image URL using a pre-trained model.

    Args:
        dataset (list): A list of numpy.ndarrays containing the image data.

    Returns:
        list: A list of tuples, where each tuple contains a mushroom species name and its corresponding probability score.
    """
    # Predict
    predictions = []
    for data in dataset:
        preds = model.predict(data)
        preds = topk(preds, 3)
        predictions.append(preds)
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
        top_confidence = first[0]
        if top_confidence >= upper_lim:
            message = f"Your fun guy is {class_d[first[1]]} with {int(top_confidence*100)}% confidence!"
        elif top_confidence >= middle_lim:
            message = (
                f"Your fun guy is probably {class_d[first[1]]} with {int(top_confidence*100)}% confidence!\n"
                + f"2nd choice: {class_d[second[1]]} with {second[0]*100:.2f}% confidence."
            )
        elif top_confidence >= lower_lim:
            message = (
                f"Im not too sure about this one, it might be {class_d[first[1]]} with {int(top_confidence*100)}% confidence, \n"
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

    return message


@app.post("/sms")
async def sms(request: Request):
    """This function receives an incoming SMS message via Twilio and sends an automated response.

    Args:
        request (Request): The incoming request object

    Returns:
        Response: The TwiML response to send back to Twilio
    """
    try:
        # Get the incoming message body and sender's phone number, needs await to work
        data = await request.form()
        sender = data["From"]
        url = data["MediaUrl0"]

        dataset = preprocessing.load_dataset([url], sender, IMAGE_SIZE)
        predictions = model_predictions(dataset)
        msg = evaluate_preds(predictions[0])

        # Create a TwiML response object
        twiml = MessagingResponse()

        # Add a message to the response
        twiml.message(msg)
        # Return the TwiML response as a string
        return Response(content=str(twiml), media_type="text/xml")

    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        # Return an empty response
        return ""


if __name__ == "__main__":
    uvicorn.run(app, port=environ.get("PORT", 8080))

import sys
from pathlib import Path

import tensorflow as tf
import pickle
from loguru import logger
from dotenv import load_dotenv
from os import environ

root = Path(__file__).parent.parent
sys.path.append(str(root))
load_dotenv()

from training.src.models.swintransformer import SwinTransformer

names = [
    "swin_tiny_224",
    "swin_small_224",
    "swin_base_224",
    "swin_base_384",
    "swin_large_224",
    "swin_large_384",
]

class_dict = pickle.load(open(str(root / "training" / "src" / "class_dict.pkl"), 'rb'))


def resave_base_model_weights(name="swin_large_224"):
    """Resave the weights of an ImageNet pre-trained Swin Transformer model as a Tensorflow SavedModel.

    Args:
        name (str): The name of the pre-trained Swin Transformer model to resave. 
                    Defaults to "swin_large_224".
    """
    image_size = int(name.split("_")[-1])

    img_adjust_layer = tf.keras.layers.Lambda(
        lambda data: tf.keras.applications.imagenet_utils.preprocess_input(
            tf.cast(data, tf.float32), mode="torch"),
        input_shape=[image_size, image_size, 3])
    pretrained_model = SwinTransformer(
        model_name=name, include_top=False, pretrained=True, root=str(root)
    )

    model = tf.keras.Sequential([
        img_adjust_layer,
        pretrained_model,
        tf.keras.layers.Dense(len(class_dict), activation='softmax')
    ])

    # model.save(str(root / "training" / "base_models" / name))  # local save (no TPU/TPU-vm)
    model.save(f'{environ["GCS_PATH"]}/{environ["GCS_BASE_MODELS"]}/{name}')  # GCS save (TPU/TPU-vm)

    logger.info(f"Saved model {name}")


for name in names:
    resave_base_model_weights(name)

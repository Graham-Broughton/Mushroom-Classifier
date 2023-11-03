import tensorflow as tf
from pathlib import Path
import sys

root = Path(__file__).parent.parent
sys.path.append(str(root))

from training.src.models.swintransformer import SwinTransformer

names = ["swin_tiny_224", "swin_small_224", "swin_base_224", "swin_base_384", "swin_large_224", "swin_large_384"]

def resave_base_model_weights(name="swin_large_224"):
    pretrained_model = SwinTransformer(name=name, include_top=False, pretrained=True, root=str(root))
    model = tf.keras.Sequential([pretrained_model])
    model.save(str(root / "training" / "base_models" / name))
    del model

for name in names:
    resave_base_model_weights(name)


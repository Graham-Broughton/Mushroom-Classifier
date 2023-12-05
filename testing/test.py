import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

logger.remove(0)


def test(CFG, files_test, model, fold):
    # PREDICT TEST USING TTA
    files_test = np.sort(
        np.array(tf.io.gfile.glob(f'{os.environ["GCS_PATH"][fold]}/test*.tfrec'))
    )    
    logger.info("Predicting Test with TTA...")
    ds_test = get_dataset(
        files_test,
        CFG,
        labeled=False,
        return_image_names=False,
        augment=True,
        repeat=True,
        shuffle=False,
        dim=CFG.IMG_SIZES[fold],
        batch_size=CFG.BATCH_SIZES[fold] * 4,
    )
    ct_test = count_data_items(files_test)
    STEPS = CFG.TTA * ct_test / CFG.BATCH_SIZES[fold] / 4 / CFG.REPLICAS
    pred = model.predict(ds_test, steps=STEPS, verbose=CFG.VERBOSE)[: CFG.TTA * ct_test,]
    preds[:, 0] += np.mean(pred.reshape((ct_test, CFG.TTA), order='F'), axis=1) * CFG.WGTS[fold]
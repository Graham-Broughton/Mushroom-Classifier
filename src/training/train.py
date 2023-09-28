# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit
from sklearn.model_selection import KFold
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import roc_auc_score
import os
from loguru import logger

from config import CFG
from src.visuals.training_viz import plot_training
from src.training.dataset import get_dataset
from src.training.utils import count_data_items, tpu_test

CFG = CFG()

# logger.remove(0)

GCS_PATH = os.environ['GCS_PATH']
AUTO = tf.data.experimental.AUTOTUNE


def build_model(model, num_classes, dim=128):
    inp = tf.keras.layers.Input(shape=(dim,dim,3))
    load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    base = hub.KerasLayer(
        model,
        trainable=True,
        load_options=load_locally
    )#(input_shape=(dim,dim,3),weights='imagenet',include_top=False)
    x = base(inp, training=False)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])
    return model
    

def sv(img_size):
    return tf.keras.callbacks.ModelCheckpoint(
        f'gs://mush-img-repo/model/swin-{img_size}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        save_freq='epoch',
    )


def get_lr_callback(batch_size=8):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * CFG.REPLICAS * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback
    

def get_history(model, files_train, files_valid, CFG):
    logger.info("Training...")
    
    history = model.fit(
        get_dataset(
            files_train, CFG, augment=True, shuffle=True, repeat=True, dim=CFG.IMG_SIZES, batch_size=CFG.BATCH_SIZES
        ),
        epochs=CFG.EPOCHS,
        callbacks=[sv(CFG.IMG_SIZES), get_lr_callback(CFG.BATCH_SIZES)],
        steps_per_epoch=count_data_items(files_train) / CFG.BATCH_SIZES // CFG.REPLICAS,
        validation_data=get_dataset(
            files_valid, CFG, augment=False, shuffle=False, repeat=False, dim=CFG.IMG_SIZES
        ),  # class_weight = {0:1,1:2},
        verbose=CFG.VERBOSE,
    )
    return history.history


def train(CFG, strategy):
    skf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_pred = []
    oof_tar = []
    oof_val = []
    oof_names = []
    oof_folds = []
    # preds = np.zeros((count_data_items(files_test), 1))

    for (idxT, idxV) in skf.split(np.arange(107)):
        print('#' * 25)
        print(
            f'#### Image Size {CFG.IMG_SIZES} with {CFG.MODELS} and batch_size {CFG.BATCH_SIZES * CFG.REPLICAS}'
        )
        logger.info(
            f"# Image Size {CFG.IMG_SIZES} with Model {CFG.MODELS} and batch_sz {CFG.BATCH_SIZES*CFG.REPLICAS}"
        )

        # CREATE TRAIN AND VALIDATION SUBSETS
        files_train = tf.io.gfile.glob([f'{GCS_PATH}/tfrec/train{x:02d}*.tfrec' for x in idxT])
        np.random.shuffle(files_train)
        print('#' * 25)
        files_valid = tf.io.gfile.glob([f'{GCS_PATH}/tfrec/train{x:02d}*.tfrec' for x in idxV])
        files_test = tf.io.gfile.glob(f'{GCS_PATH}/tfrec/val*.tfrec')

        # BUILD MODEL
        K.clear_session()
        with strategy.scope():
            model = build_model(CFG.MODELS, CFG.NUM_CLASSES, dim=CFG.IMG_SIZES)

        # TRAIN
        history = get_history(model, files_train, files_valid, CFG)

        logger.info("Loading best model...")
        model.load_weights(f'{GCS_PATH}/models/swin-{CFG.IMG_SIZES}.h5') 
        # PREDICT OOF USING TTA
        logger.info("Predicting OOF with TTA...")
        ds_valid = get_dataset(
            files_valid,
            CFG,
            labeled=False,
            return_image_names=False,
            augment=True,
            repeat=True,
            shuffle=False,
            dim=CFG.IMG_SIZES,
            batch_size=CFG.BATCH_SIZES * 4,
        )
        ct_valid = count_data_items(files_valid)
        STEPS = CFG.TTA * ct_valid / CFG.BATCH_SIZES / 4 / CFG.REPLICAS
        pred = model.predict(ds_valid, steps=STEPS, verbose=CFG.VERBOSE)[: CFG.TTA * ct_valid,]
        oof_pred.append(np.mean(pred.reshape((ct_valid, CFG.TTA), order='F'), axis=1))
        # oof_pred.append(model.predict(get_dataset(files_valid,dim=CFG.IMG_SIZES),verbose=1))

        # GET OOF TARGETS AND NAMES
        ds_valid = get_dataset(
            files_valid, CFG, 
            augment=False, 
            repeat=False, 
            dim=CFG.IMG_SIZES, 
            labeled=True, 
            return_image_names=True
        )
        oof_tar.append(np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]))
        ds = get_dataset(
            files_valid, CFG, 
            augment=False, 
            repeat=False, 
            dim=CFG.IMG_SIZES, 
            labeled=False, 
            return_image_names=True
        )
        oof_names.append(np.array([img_name.numpy().decode("utf-8") for img, img_name in iter(ds.unbatch())]))

        # REPORT RESULTS
        auc = roc_auc_score(oof_tar[-1], oof_pred[-1])
        oof_val.append(np.max(history.history['val_auc']))
        logger.info(f"#### OOF AUC without TTA = {oof_val[-1]}, with TTA = {auc}")

        # PLOT TRAINING
        if CFG.DISPLAY_PLOT:
            plot_training(history, CFG)


if __name__ == '__main__':
    strategy, tpu = tpu_test(CFG)
    train(CFG, strategy)
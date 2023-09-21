# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit
from dotenv import load_dotenv, set_key
from sklearn.model_selection import KFold
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.metrics import roc_auc_score
import os
from loguru import logger

from config import CFG
from src.visuals.training_viz import plot_training
from dataset import get_dataset
from utils import count_data_items, tpu_test

CFG = CFG()
load_dotenv()

logger.remove(0)

GCS_PATH = os.environ['GCS_PATH']
AUTO = tf.data.experimental.AUTOTUNE


def build_model(CFG, dim=128, ef=0):
    

def sv(fold):
    return tf.keras.callbacks.ModelCheckpoint(
        'fold-%i.h5' % fold,
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
    

def get_history(model, fold, files_train, files_valid, CFG):
    logger.info("Training...")
    history = model.fit(
        get_dataset(
            files_train, CFG, augment=True, shuffle=True, repeat=True, dim=CFG.IMG_SIZES[fold], batch_size=CFG.BATCH_SIZES[fold]
        ),
        epochs=CFG.EPOCHS[fold],
        callbacks=[sv(fold), get_lr_callback(CFG.BATCH_SIZES[fold])],
        steps_per_epoch=count_data_items(files_train) / CFG.BATCH_SIZES[fold] // CFG.REPLICAS,
        validation_data=get_dataset(
            files_valid, CFG, augment=False, shuffle=False, repeat=False, dim=CFG.IMG_SIZES[fold]
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

    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):
        # DISPLAY FOLD INFO
        print('#' * 25)
        print('#### FOLD', fold + 1)
        print(
            '#### Image Size %i with EfficientNet B%i and batch_size %i'
            % (CFG.IMG_SIZES[fold], CFG.MODELS[fold], CFG.BATCH_SIZES[fold] * CFG.REPLICAS)
        )
        logger.info(
            f"# Image Size {CFG.IMG_SIZES[fold]} with Model {CFG.MODELS[fold]} and batch_sz {CFG.BATCH_SIZES[fold]*CFG.REPLICAS}"
        )

        # CREATE TRAIN AND VALIDATION SUBSETS
        files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec' % x for x in idxT])
        np.random.shuffle(files_train)
        print('#' * 25)
        files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec' % x for x in idxV])

        # BUILD MODEL
        K.clear_session()
        with strategy.scope():
            model = build_model(dim=CFG.IMG_SIZES[fold], ef=CFG.MODELS[fold])


        # TRAIN
        history = get_history(model)

        logger.info("Loading best model...")
        model.load_weights('fold-%i.h5' % fold)

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
            dim=CFG.IMG_SIZES[fold],
            batch_size=CFG.BATCH_SIZES[fold] * 4,
        )
        ct_valid = count_data_items(files_valid)
        STEPS = CFG.TTA * ct_valid / CFG.BATCH_SIZES[fold] / 4 / CFG.REPLICAS
        pred = model.predict(ds_valid, steps=STEPS, verbose=CFG.VERBOSE)[: CFG.TTA * ct_valid,]
        oof_pred.append(np.mean(pred.reshape((ct_valid, CFG.TTA), order='F'), axis=1))
        # oof_pred.append(model.predict(get_dataset(files_valid,dim=CFG.IMG_SIZES[fold]),verbose=1))

        # GET OOF TARGETS AND NAMES
        ds_valid = get_dataset(
            files_valid, CFG, 
            augment=False, 
            repeat=False, 
            dim=CFG.IMG_SIZES[fold], 
            labeled=True, 
            return_image_names=True
        )
        oof_tar.append(np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]))
        oof_folds.append(np.ones_like(oof_tar[-1], dtype='int8') * fold)
        ds = get_dataset(
            files_valid, CFG, 
            augment=False, 
            repeat=False, 
            dim=CFG.IMG_SIZES[fold], 
            labeled=False, 
            return_image_names=True
        )
        oof_names.append(np.array([img_name.numpy().decode("utf-8") for img, img_name in iter(ds.unbatch())]))

        # REPORT RESULTS
        auc = roc_auc_score(oof_tar[-1], oof_pred[-1])
        oof_val.append(np.max(history.history['val_auc']))
        logger.info(f"#### FOLD {fold + 1} OOF AUC without TTA = {oof_val[-1]}, with TTA = {auc}")

        # PLOT TRAINING
        if CFG.DISPLAY_PLOT:
            plot_training(history, fold, CFG)

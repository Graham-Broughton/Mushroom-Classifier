import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import math
AUTOTUNE = tf.data.experimental.AUTOTUNE

ROT_ = 180.0
SHR_ = 2.0
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0

def parse_tfrecord(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example['image'], example['category_id']

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                   -s1,  c1,   zero, 
                                   zero, zero, one])    
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    return K.dot(K.dot(rotation_matrix, shear_matrix), 
                 K.dot(zoom_matrix,     shift_matrix))

def transform_2(image, DIM):
    """input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    output - image randomly rotated, sheared, zoomed, and shifted"""
    XDIM = DIM%2 #fix for size 331
    rot = ROT_ * tf.random.normal([1])          #, dtype='float32'
    shr = SHR_ * tf.random.normal([1]) #, dtype='float32'
    h_zoom = 1.0 + tf.random.normal([1]) / HZOOM_#, dtype='float32'
    w_zoom = 1.0 + tf.random.normal([1]) / WZOOM_#, dtype='float32'
    h_shift = HSHIFT_ * tf.random.normal([1]) #, dtype='float32'
    w_shift = WSHIFT_ * tf.random.normal([1]) #, dtype='float32'

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype=tf.int32)
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype=tf.float32))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d,[DIM, DIM,3])

def _random_crop(image, DIM):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        area_range=(0.08, 1.0),
        aspect_ratio_range=(0.75, 1.33),
        max_attempts=100,
        min_object_covered=0.1,
    )
    image = tf.slice(image, begin, size)
    image = tf.image.resize(image, size=[DIM, DIM])
    return image

def flips(image):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if p_spatial > 0.75:
        image = tf.image.transpose(image)
    return image

def rotates(image):
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    if p_rotate > 0.66:
        image = tf.image.rot90(image, k=3)
    elif p_rotate > 0.33:
        image = tf.image.rot90(image, k=2)
    else:
        image = tf.image.rot90(image, k=1)
    return image

def colour(image):
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    return image

def data_augment(image, DIM):
    image = _random_crop(image, DIM)
    image = transform_2(image, DIM)
    image = rotates(image)
    image = flips(image)
    image = colour(image)
    image = tf.clip_by_value(image, 0, 255)
    return image

def get_dataset(filenames, batch_size, DIM, augment=True):
    opt = tf.data.Options()
    opt.deterministic = False

    ds = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .repeat()
        .shuffle(batch_size * 10)
        .with_options(opt)
        .map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    )
    if augment==True:
        ds = ds.map(lambda image, label: (data_augment(image, DIM), label), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda image, label: (tf.data.resize(image, size=[DIM, DIM]), label), num_parellel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
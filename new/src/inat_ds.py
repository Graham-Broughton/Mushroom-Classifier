def load_img_inat(filepath, size):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(img, 3)
    img = tf.cast(img, tf.float32)
    img = applications.efficientnet_v2.preprocess_input(img)
    img = tf.image.resize(img, size=size)
    return img
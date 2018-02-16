from network import Model
import tensorflow as tf
import numpy as np
import logger
from os.path import basename, splitext
import matplotlib.pyplot as plt

EPOCHS = 1
BATCHES = 1
filenames = tf.constant(["test_images/macula.jpg"], dtype=tf.string)
labels = tf.constant([1], dtype=tf.float32)

logger.configure('/tmp')
sess = tf.Session()
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [256, 256])
    image_normal = tf.image.per_image_standardization(image_resized)
    image_ran_lr = tf.image.random_flip_left_right(image_normal)
    image_ran_ud = tf.image.random_flip_up_down(image_ran_lr)
    image_expand_dims = tf.reshape(image_ran_ud, shape=[1, -1])
    return image_ran_ud, label

model = Model(BATCHES, 'weighted')

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.prefetch(100)
dataset = dataset.map(_parse_function, 10)
dataset = dataset.shuffle(100)
dataset = dataset.repeat(EPOCHS)
dataset = dataset.batch(BATCHES)

itere = dataset.make_one_shot_iterator()
ele = itere.get_next()
print(ele)
model.build(ele[0], ele[1])

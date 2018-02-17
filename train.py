import tensorflow as tf

import logger
from network import Model

EPOCHS = 1
BATCHES = 32
filenames = tf.constant(["test_images/macula.jpg"], dtype = tf.string)
labels = tf.constant([1], dtype = tf.float32)

logger.configure('/tmp')
sess = tf.Session()


def _build_dataset(_filenames, _labels, epochs, batches):
	dataset = tf.data.Dataset.from_tensor_slices((_filenames, _labels))
	dataset = dataset.prefetch(100)
	dataset = dataset.map(_parse_function, 10)
	dataset = dataset.shuffle(100)
	dataset = dataset.repeat(epochs)
	dataset = dataset.batch(batches)
	return dataset


def _parse_function(filename, label):
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string, channels = 3)
	image_resized = tf.image.resize_images(image_decoded, [256, 256])
	image_normal = tf.image.per_image_standardization(image_resized)
	image_ran_lr = tf.image.random_flip_left_right(image_normal)
	image_ran_ud = tf.image.random_flip_up_down(image_ran_lr)
	image_expand = tf.expand_dims(image_ran_ud, 0)
	patches = tf.extract_image_patches(image_expand, [1, 32, 32, 1], [1, 32, 32, 1], [1, 1, 1, 1], 'SAME')[0]
	patches = tf.reshape(patches, [-1, 32, 32, 3])
	return patches, label


train_data = _build_dataset(filenames, labels, EPOCHS, BATCHES)
iterator = train_data.make_one_shot_iterator()

model = Model(BATCHES, 'weighted')
model.build(*iterator.get_next())

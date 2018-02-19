import math

import tensorflow as tf

import logger
from network import Model

filenames = tf.constant(["test_images/macula.jpg"], dtype = tf.string)
labels = tf.constant([1], dtype = tf.float32)

val_filenames = tf.constant(["test_images/lena.jpg"], dtype = tf.string)
val_labels = tf.constant([1], dtype = tf.float32)

MIN_VAL = math.inf
EPOCHS = 100
BATCHES = 1
NO_OF_ITERS = int(filenames.get_shape()[0]) // BATCHES
LOG_DIR = '/tmp'
SAVE_DIR = '/tmp/macula-iqa.cpkt'
LEARNING_RATE = 1e-3
DROPOUT_PROB = 0.5

logger.configure(LOG_DIR)
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
validation_data = _build_dataset(val_filenames, val_labels, 1, int(val_filenames.get_shape()[0]))

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(train_data)
validation_init_op = iterator.make_initializer(validation_data)

model = Model('weighted')
model.build(*next_element)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

training_feed = {model.lr: LEARNING_RATE, model.prob: DROPOUT_PROB}
testing_feed = {model.prob: DROPOUT_PROB}

for epoch in range(EPOCHS):
	sess.run(training_init_op)
	l_avg = 0
	for iters in range(NO_OF_ITERS):
		_, train_loss = sess.run([model.train_op, model.loss_op], feed_dict = training_feed)
		l_avg += train_loss
		logger.record_tabular('training_loss', train_loss)

	l_avg /= NO_OF_ITERS

	sess.run(validation_init_op)
	val_loss = sess.run([model.loss_op], feed_dict = testing_feed)
	val_loss = val_loss[0]
	print("Epoch", epoch, "\t Training loss:", l_avg, "\t Validation loss:", val_loss)
	logger.record_tabular('avg_training_loss', l_avg)
	logger.record_tabular('validation_loss', val_loss)
	logger.dump_tabular()

	if val_loss < MIN_VAL:
		MIN_VAL = val_loss
		saver.save(sess, SAVE_DIR)
		print("Savig model")

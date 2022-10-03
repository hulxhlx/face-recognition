from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import facenet
import h5py
import math
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import loaddata
image_size=160

logs_base_dir='./logs'
models_base_dir= './models'
data_dir='E:/anaconda/envs/tfface/Lib/site-packages/facenet/datasets/lfw_mtcnnpy_160'
lfw_dir = 'E:/anaconda/envs/tfface/Lib/site-packages/facenet/datasets/lfw_mtcnnpy_160'
lfw_pairs = 'E:/anaconda/envs/tfface/Lib/site-packages/facenet/data/pairs.txt'
optimizer = 'RMSPROP'
arglearning_rate=0
max_nrof_epochs = 30
batch_size = 64
keep_probability=0.8
weight_decay = 5e-5
center_loss_factor=1e-2
center_loss_alfa = 0.9
seed=666
filter_filename = ''
filter_min_nrof_images_per_class=100
filter_percentile=''
validation_set_split_ratio=0
min_nrof_val_images_per_class = 0
# if_pretrained_model='E:/anaconda/envs/tfface/Lib/site-packages/facenet/models/20180408-102900'
if_pretrained_model=''

epoch_size=1000
embedding_size=128
keep_probability=1.0
prelogits_norm_p=1.0
prelogits_norm_loss_factor=0.0
center_loss_alfa=0.95
center_loss_factor=0.0
moving_average_decay=0.9999
learning_rate_decay_epochs=100
learning_rate_decay_factor=1.0
log_histograms='store_true'
max_nrof_epochs=500
validate_every_n_epochs=5
learning_rate_schedule_file='E:/anaconda/envs/tfface/Lib/site-packages/facenet/data/learning_rate_retrain_tripletloss.txt'
gpu_memory_fraction=1.0
random_rotate=True
random_crop=True
random_flip=True
prelogits_norm=1.0
prelogits_hist_max=10.0
use_fixed_image_standardization=False
lfw_distance_metric=0
lfw_subtract_mean='store_true'
lfw_use_flipped_images='store_true'
lfw_batch_size=100
lfw_nrof_folds=10

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
	# Calculate evaluation metrics
	thresholds = np.arange(0, 4, 0.01)
	embeddings1 = embeddings[0::2]
	embeddings2 = embeddings[1::2]
	tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
	                                               np.asarray(actual_issame), nrof_folds=nrof_folds,
	                                               distance_metric=distance_metric, subtract_mean=subtract_mean)
	thresholds = np.arange(0, 4, 0.001)
	val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
	                                              np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds,
	                                              distance_metric=distance_metric, subtract_mean=subtract_mean)
	return tpr, fpr, accuracy, val, val_std, far


def get_paths(lfw_dir, pairs):
	nrof_skipped_pairs = 0
	path_list = []
	issame_list = []
	for pair in pairs:
		if len(pair) == 3:
			path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
			path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
			issame = True
		elif len(pair) == 4:
			path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
			path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
			issame = False
		if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
			path_list += (path0, path1)
			issame_list.append(issame)
		else:
			nrof_skipped_pairs += 1
	if nrof_skipped_pairs > 0:
		print('Skipped %d image pairs' % nrof_skipped_pairs)

	return path_list, issame_list


def add_extension(path):
	if os.path.exists(path + '.jpg'):
		return path + '.jpg'
	elif os.path.exists(path + '.png'):
		return path + '.png'
	else:
		raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def read_pairs(pairs_filename):
	pairs = []
	with open(pairs_filename, 'r') as f:
		for line in f.readlines()[1:]:
			pair = line.strip().split()
			pairs.append(pair)
	return np.array(pairs)


def find_threshold(var, percentile):
	hist, bin_edges = np.histogram(var, 100)
	cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
	# plt.plot(bin_centers, cdf)
	threshold = np.interp(percentile * 0.01, cdf, bin_centers)
	return threshold


def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
	with h5py.File(data_filename, 'r') as f:
		distance_to_center = np.array(f.get('distance_to_center'))
		label_list = np.array(f.get('label_list'))
		image_list = np.array(f.get('image_list'))
		distance_to_center_threshold = find_threshold(distance_to_center, percentile)
		indices = np.where(distance_to_center >= distance_to_center_threshold)[0]
		filtered_dataset = dataset
		removelist = []
		for i in indices:
			label = label_list[i]
			image = image_list[i]
			if image in filtered_dataset[label].image_paths:
				filtered_dataset[label].image_paths.remove(image)
			if len(filtered_dataset[label].image_paths) < min_nrof_images_per_class:
				removelist.append(label)

		ix = sorted(list(set(removelist)), reverse=True)
		for i in ix:
			del (filtered_dataset[i])

	return filtered_dataset


def train(arglearning_rate, epoch_size, sess, epoch, image_list, label_list, image_batch, label_batch,
          image_paths_placeholder, labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step,
          loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file,
          stat, cross_entropy_mean, accuracy,
          learning_rate, prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm,
          prelogits_hist_max, use_fixed_image_standardization):
	batch_number = 0

	if arglearning_rate > 0.0:
		lr = arglearning_rate
	else:
		lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

	if lr <= 0:
		return False


	dataset = loaddata.read_data_sets('E:/anaconda/envs/tfface/Lib/site-packages/facenet/datasets/lfw_mtcnnpy_160')

	train_data=dataset.train
	# Training loop
	train_time = 0
	while batch_number < epoch_size:
		start_time = time.time()
		tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm,
		               accuracy, prelogits_center_loss]
		if batch_number % 100 == 0:
			batch_features, batch_labels = train_data.next_batch(64)
			print(sess.run(image_batch, feed_dict={image_batch :batch_features})[3][5][5])
			print('run')


		duration = time.time() - start_time

		batch_number += 1
		train_time += duration

	return True


def validate(lfw_batch_size, sess, epoch, image_list, label_list, enqueue_op, image_paths_placeholder,
             labels_placeholder, control_placeholder,
             phase_train_placeholder, batch_size_placeholder,
             stat, loss, regularization_losses, cross_entropy_mean, accuracy, validate_every_n_epochs,
             use_fixed_image_standardization):
	print('Running forward pass on validation set')

	nrof_batches = len(label_list) // lfw_batch_size
	nrof_images = nrof_batches * lfw_batch_size

	# Enqueue one epoch of image paths and labels
	labels_array = np.expand_dims(np.array(label_list[:nrof_images]), 1)
	image_paths_array = np.expand_dims(np.array(image_list[:nrof_images]), 1)
	control_array = np.ones_like(labels_array,
	                             np.int32) * facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
	sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
	                      control_placeholder: control_array})

	loss_array = np.zeros((nrof_batches,), np.float32)
	xent_array = np.zeros((nrof_batches,), np.float32)
	accuracy_array = np.zeros((nrof_batches,), np.float32)

	# Training loop
	start_time = time.time()
	for i in range(nrof_batches):
		feed_dict = {phase_train_placeholder: False, batch_size_placeholder: lfw_batch_size}
		loss_, cross_entropy_mean_, accuracy_ = sess.run([loss, cross_entropy_mean, accuracy], feed_dict=feed_dict)
		loss_array[i], xent_array[i], accuracy_array[i] = (loss_, cross_entropy_mean_, accuracy_)
		if i % 10 == 9:
			print('.', end='')
			sys.stdout.flush()
	print('')

	duration = time.time() - start_time

	val_index = (epoch - 1) // validate_every_n_epochs
	stat['val_loss'][val_index] = np.mean(loss_array)
	stat['val_xent_loss'][val_index] = np.mean(xent_array)
	stat['val_accuracy'][val_index] = np.mean(accuracy_array)

	print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f' %
	      (epoch, duration, np.mean(loss_array), np.mean(xent_array), np.mean(accuracy_array)))


def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
             batch_size_placeholder, control_placeholder,
             embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer,
             stat, epoch, distance_metric, subtract_mean, use_flipped_images, use_fixed_image_standardization):
	start_time = time.time()
	# Run forward pass to calculate embeddings
	print('Runnning forward pass on LFW images')

	# Enqueue one epoch of image paths and labels
	nrof_embeddings = len(actual_issame) * 2  # nrof_pairs * nrof_images_per_pair
	nrof_flips = 2 if use_flipped_images else 1
	nrof_images = nrof_embeddings * nrof_flips
	labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
	image_paths_array = np.expand_dims(np.repeat(np.array(image_paths), nrof_flips), 1)
	control_array = np.zeros_like(labels_array, np.int32)
	if use_fixed_image_standardization:
		control_array += np.ones_like(labels_array) * facenet.FIXED_STANDARDIZATION
	if use_flipped_images:
		# Flip every second image
		control_array += (labels_array % 2) * facenet.FLIP
	sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
	                      control_placeholder: control_array})

	embedding_size = int(embeddings.get_shape()[1])
	assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
	nrof_batches = nrof_images // batch_size
	emb_array = np.zeros((nrof_images, embedding_size))
	lab_array = np.zeros((nrof_images,))
	for i in range(nrof_batches):
		feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
		emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
		lab_array[lab] = lab
		emb_array[lab, :] = emb
		if i % 10 == 9:
			print('.', end='')
			sys.stdout.flush()
	print('')
	embeddings = np.zeros((nrof_embeddings, embedding_size * nrof_flips))
	if use_flipped_images:
		# Concatenate embeddings for flipped and non flipped version of the images
		embeddings[:, :embedding_size] = emb_array[0::2, :]
		embeddings[:, embedding_size:] = emb_array[1::2, :]
	else:
		embeddings = emb_array

	assert np.array_equal(lab_array, np.arange(
		nrof_images)) == True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
	_, _, accuracy, val, val_std, far = evaluate(embeddings, actual_issame, nrof_folds=nrof_folds,
	                                                 distance_metric=distance_metric, subtract_mean=subtract_mean)

	print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
	print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
	lfw_time = time.time() - start_time
	# Add validation loss and accuracy to summary
	summary = tf.Summary()
	# pylint: disable=maybe-no-member
	summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
	summary.value.add(tag='lfw/val_rate', simple_value=val)
	summary.value.add(tag='time/lfw', simple_value=lfw_time)
	summary_writer.add_summary(summary, step)
	with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
		f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
	stat['lfw_accuracy'][epoch - 1] = np.mean(accuracy)
	stat['lfw_valrate'][epoch - 1] = val


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
	# Save the model checkpoint
	print('Saving variables')
	start_time = time.time()
	checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
	saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
	save_time_variables = time.time() - start_time
	print('Variables saved in %.2f seconds' % save_time_variables)
	metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
	save_time_metagraph = 0
	if not os.path.exists(metagraph_filename):
		print('Saving metagraph')
		start_time = time.time()
		saver.export_meta_graph(metagraph_filename)
		save_time_metagraph = time.time() - start_time
		print('Metagraph saved in %.2f seconds' % save_time_metagraph)
	summary = tf.Summary()
	# pylint: disable=maybe-no-member
	summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
	summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
	summary_writer.add_summary(summary, step)


network = importlib.import_module('models.inception_resnet_v1')
print(network)
image_size = (image_size, image_size)
subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
log_dir = os.path.join(os.path.expanduser(logs_base_dir), subdir)
if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
	os.makedirs(log_dir)
model_dir = os.path.join(os.path.expanduser(models_base_dir), subdir)
if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
	os.makedirs(model_dir)

stat_file_name = os.path.join(log_dir, 'stat.h5')

# Write arguments to a text file
# facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

# Store some git revision info in a text file in the log directory
# src_path,_ = os.path.split(os.path.realpath(__file__))
# facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

np.random.seed(seed=seed)
random.seed(seed)
dataset = facenet.get_dataset(data_dir)
print(dataset)
# if filter_filename:
#     dataset = filter_dataset(dataset, os.path.expanduser(filter_filename),
#         filter_percentile, filter_min_nrof_images_per_class)

if validation_set_split_ratio > 0.0:
	train_set, val_set = facenet.split_dataset(dataset, validation_set_split_ratio, min_nrof_val_images_per_class,
	                                           'SPLIT_IMAGES')
else:
	train_set, val_set = dataset, []

nrof_classes = len(train_set)
print(nrof_classes)
print('Model directory: %s' % model_dir)
print('Log directory: %s' % log_dir)
pretrained_model = None
if if_pretrained_model:
	pretrained_model = os.path.expanduser(if_pretrained_model)
	print('Pre-trained model: %s' % pretrained_model)

if lfw_dir:
	print('LFW directory: %s' % lfw_dir)
	# Read the file containing the pairs used for testing
	pairs = read_pairs(os.path.expanduser(lfw_pairs))
	# Get the paths for the corresponding images
	lfw_paths, actual_issame = get_paths(os.path.expanduser(lfw_dir), pairs)

with tf.Graph().as_default():
	tf.set_random_seed(seed)
	global_step = tf.Variable(0, trainable=False)

	# Get a list of image paths and their labels
	image_list, label_list = facenet.get_image_paths_and_labels(train_set)

	assert len(image_list) > 0, 'The training set should not be empty'

	val_image_list, val_label_list = facenet.get_image_paths_and_labels(val_set)

	# Create a queue that produces indices into the image_list and label_list
	labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
	range_size = array_ops.shape(labels)[0]
	learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
	batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
	phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
	image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
	labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
	control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')

	nrof_preprocess_threads = 4



	image_batch =tf.placeholder(
                    dtype=tf.float32, shape=[None,160,160,3], name="x_input"
                )
	label_batch = tf.placeholder(
		dtype=tf.int32, shape=[None, ], name="actual_label"
	)	# Tensor("batch_join:0", shape=(?, 160, 160, 3), dtype=float32)
	# Tensor("batch_join:1", shape=(?,), dtype=int32)
	# (?,)
	print('Number of classes in training set: %d' % nrof_classes)
	print('Number of examples in training set: %d' % len(image_list))

	print('Number of classes in validation set: %d' % len(val_set))
	print('Number of examples in validation set: %d' % len(val_image_list))

	print('Building training graph')
	#
	# # Build the inference graph
	prelogits, _ = network.inference(image_batch, keep_probability,
	                                 phase_train=phase_train_placeholder, bottleneck_layer_size=embedding_size,
	                                 weight_decay=weight_decay)
	logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
	                              weights_initializer=slim.initializers.xavier_initializer(),
	                              weights_regularizer=slim.l2_regularizer(weight_decay),
	                              scope='Logits', reuse=False)

	embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

	# Norm for the prelogits
	eps = 1e-4
	prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=prelogits_norm_p, axis=1))
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * prelogits_norm_loss_factor)

	# Add center loss
	prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, center_loss_alfa, nrof_classes)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * center_loss_factor)

	learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
	                                           learning_rate_decay_epochs * epoch_size, learning_rate_decay_factor,
	                                           staircase=True)
	tf.summary.scalar('learning_rate', learning_rate)

	# Calculate the average cross entropy loss across the batch
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=label_batch, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
	accuracy = tf.reduce_mean(correct_prediction)

	# Calculate the total losses
	regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

	# Build a Graph that trains the model with one batch of examples and updates the model parameters
	train_op = facenet.train(total_loss, global_step, optimizer,
	                             learning_rate, moving_average_decay, tf.global_variables(), log_histograms)

	# Create a saver
	saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

	# Build the summary operation based on the TF collection of Summaries.
	summary_op = tf.summary.merge_all()

	# Start running operations on the Graph.
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
	coord = tf.train.Coordinator()
	tf.train.start_queue_runners(coord=coord, sess=sess)

	with sess.as_default():

		if pretrained_model:
			print('Restoring pretrained model: %s' % pretrained_model)
			saver.restore(sess, pretrained_model)

		# Training and validation loop
		print('Running training')
		nrof_steps = max_nrof_epochs * epoch_size
		nrof_val_samples = int(math.ceil(
			max_nrof_epochs / validate_every_n_epochs))  # Validate every validate_every_n_epochs as well as in the last epoch
		stat = {
			'loss': np.zeros((nrof_steps,), np.float32),
			'center_loss': np.zeros((nrof_steps,), np.float32),
			'reg_loss': np.zeros((nrof_steps,), np.float32),
			'xent_loss': np.zeros((nrof_steps,), np.float32),
			'prelogits_norm': np.zeros((nrof_steps,), np.float32),
			'accuracy': np.zeros((nrof_steps,), np.float32),
			'val_loss': np.zeros((nrof_val_samples,), np.float32),
			'val_xent_loss': np.zeros((nrof_val_samples,), np.float32),
			'val_accuracy': np.zeros((nrof_val_samples,), np.float32),
			'lfw_accuracy': np.zeros((max_nrof_epochs,), np.float32),
			'lfw_valrate': np.zeros((max_nrof_epochs,), np.float32),
			'learning_rate': np.zeros((max_nrof_epochs,), np.float32),
			'time_train': np.zeros((max_nrof_epochs,), np.float32),
			'time_validate': np.zeros((max_nrof_epochs,), np.float32),
			'time_evaluate': np.zeros((max_nrof_epochs,), np.float32),
			'prelogits_hist': np.zeros((max_nrof_epochs, 1000), np.float32),
		}
		for epoch in range(1, max_nrof_epochs + 1):
			step = sess.run(global_step, feed_dict=None)
			# Train for one epoch
			t = time.time()


			cont = train(arglearning_rate, epoch_size, sess, epoch, image_list, label_list, image_batch,
			             label_batch, image_paths_placeholder, labels_placeholder,
			             learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
			             control_placeholder, global_step,
			             total_loss, train_op, summary_op, summary_writer, regularization_losses,
			             learning_rate_schedule_file,
			             stat, cross_entropy_mean, accuracy, learning_rate,
			             prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm,
			             prelogits_hist_max, use_fixed_image_standardization)
			stat['time_train'][epoch - 1] = time.time() - t
			if not cont:
				break

			t = time.time()
			if len(val_image_list) > 0 and (
					(epoch - 1) % validate_every_n_epochs == validate_every_n_epochs - 1 or epoch == max_nrof_epochs):
				validate(lfw_batch_size, sess, epoch, val_image_list, val_label_list, enqueue_op,
				         image_paths_placeholder, labels_placeholder, control_placeholder,
				         phase_train_placeholder, batch_size_placeholder,
				         stat, total_loss, regularization_losses, cross_entropy_mean, accuracy, validate_every_n_epochs,
				         use_fixed_image_standardization)
			stat['time_validate'][epoch - 1] = time.time() - t

			# Save variables and the metagraph if it doesn't exist already
			save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)

			# Evaluate on LFW
			t = time.time()
			if lfw_dir:
				evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
				         batch_size_placeholder, control_placeholder,
				         embeddings, label_batch, lfw_paths, actual_issame, lfw_batch_size, lfw_nrof_folds, log_dir,
				         step, summary_writer, stat, epoch,
				         lfw_distance_metric, lfw_subtract_mean, lfw_use_flipped_images,
				         use_fixed_image_standardization)
			stat['time_evaluate'][epoch - 1] = time.time() - t

			print('Saving statistics')
			with h5py.File(stat_file_name, 'w') as f:
				for key, value in stat.iteritems():
					f.create_dataset(key, data=value)


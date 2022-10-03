# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V2 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from tensorflow import keras
import time
import sys
import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D

def Conv_layer(x, filters, kernel_size, strides=(1, 1), padding='SAME', weight_decay=0.0005, rate=0.5, drop=True):
	x = Conv2D(filters, kernel_size, strides=strides, padding=padding,
		           kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	if drop:
			x = Dropout(rate=rate)(x)
	return x
def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
	batch_norm_params = {
		# Decay for the moving averages.
		'decay': 0.995,
		# epsilon to prevent 0s in variance.
		'epsilon': 0.001,
		# force in-place updates of mean and variance estimates
		'updates_collections': None,
		# Moving averages ends up in the trainable variables collection
		'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
	}
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
	                    weights_initializer=slim.initializers.xavier_initializer(),
	                    weights_regularizer=slim.l2_regularizer(weight_decay),
	                    normalizer_fn=slim.batch_norm,
	                    normalizer_params=batch_norm_params):
		return vgg19(images, is_training=phase_train,
		                           dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size,
		                           reuse=reuse)


def vgg19(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV2'):
	"""Creates the Inception Resnet V2 model.
	Args:
	  inputs: a 4-D tensor of size [batch_size, height, width, 3].
	  num_classes: number of predicted classes.
	  is_training: whether is training or not.
	  dropout_keep_prob: float, the fraction to keep before final layer.
	  reuse: whether or not the network and its variables should be reused. To be
		able to reuse 'scope' must be given.
	  scope: Optional variable_scope.
	Returns:
	  logits: the logits outputs of the model.
	  end_points: the set of end_points from the inception model.
	"""
	end_points = {}

	with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
		with slim.arg_scope([slim.batch_norm, slim.dropout],
		                    is_training=is_training):
			with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
			                    stride=1, padding='SAME'):
				# 149 x 149 x 32
				conv1_1 = slim.conv2d(inputs, 64, 3, stride=1, padding='VALID',
                                  scope='Conv2d_1a')
				conv1_2 = slim.conv2d(conv1_1, 64, 3, stride=1, padding='VALID',
                                  scope='Conv2d_1b')
				maxPooling1 = slim.max_pool2d(conv1_2, 2,  padding='VALID',
                                      scope='MaxPool_1c')
				conv2_1 = slim.conv2d(maxPooling1, 128, 3, stride=1, padding='VALID',
                                  scope='Conv2d_2a')
				conv2_2 = slim.conv2d(conv2_1, 128, 3, stride=1, padding='VALID',
				                      scope='Conv2d_2b')
				maxPooling2 = slim.max_pool2d(conv2_2, 2,  padding='VALID',
                                      scope='MaxPool_2c')
				conv3_1 = slim.conv2d(maxPooling2, 256, 3, stride=1, padding='VALID',
				                      scope='Conv2d_3a')
				conv3_2 = slim.conv2d(conv3_1, 256, 3, stride=1, padding='VALID',
				                      scope='Conv2d_3b')
				conv3_3 = slim.conv2d(conv3_2, 256, 3, stride=1, padding='VALID',
				                      scope='Conv2d_3c')
				conv3_4 = slim.conv2d(conv3_3, 256, 3, stride=1, padding='VALID',
				                      scope='Conv2d_3d')
				maxPooling3 = slim.max_pool2d(conv3_4, 2,padding='VALID',
				                              scope='MaxPool_3e')
				conv4_1 = slim.conv2d(maxPooling3, 512, 3, stride=1, padding='VALID',
				                      scope='Conv2d_4a')
				conv4_2 = slim.conv2d(conv4_1, 512, 3, stride=1, padding='VALID',
				                      scope='Conv2d_4b')
				conv4_3 = slim.conv2d(conv4_2, 512, 3, stride=1, padding='VALID',
				                      scope='Conv2d_4c')
				conv4_4 = slim.conv2d(conv4_3, 512, 3, stride=1, padding='VALID',
				                      scope='Conv2d_4d')
				maxPooling4 = slim.max_pool2d(conv4_4, 2, padding='VALID',
				                              scope='MaxPool_4e')
				conv5_1 = slim.conv2d(maxPooling4, 512, 3, stride=1, padding='VALID',
				                      scope='Conv2d_5a')
				conv5_2 = slim.conv2d(conv5_1, 512, 3, stride=1, padding='VALID',
				                      scope='Conv2d_5b')
				conv5_3 = slim.conv2d(conv5_2, 512, 3, stride=1, padding='VALID',
				                      scope='Conv2d_5c')
				conv5_4 = slim.conv2d(conv5_3, 512, 3, stride=1, padding='VALID',
				                      scope='Conv2d_5d')
				maxPooling5 = slim.max_pool2d(conv5_4, 2, stride=2, padding='VALID',
				                              scope='MaxPool_5e')
				flat = slim.flatten(maxPooling5)
				dropout =  slim.dropout(flat, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')

				net = slim.fully_connected(dropout, bottleneck_layer_size, activation_fn=None,
	                           scope='Bottleneck', reuse=False)
	return net, end_points

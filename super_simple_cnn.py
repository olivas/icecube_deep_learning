#!/usr/bin/env python

import os
import argparse
import logging
import pickle
import random
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='Hello IceCube')
parser.add_argument('--data_path',
                    default='/opt/deep_learning/data',
                    help='Path to training and test data.')
parser.add_argument('--epochs',
                    dest='epochs',
                    default=10,
                    type=int,
                    help='Number of units in the hidden layer.')
parser.add_argument('--n-hidden-units',
                    dest='n_hidden_units',
                    default=512,
                    type=int,
                    help='Number of units in the hidden layer.')
parser.add_argument('--activation',
                    dest='activation',
                    default='relu',
                    help='Name of tf.keras.activation function.')
parser.add_argument('--optimizer',
                    dest='optimizer',
                    default='sgd',
                    help='Name of tf.keras.optimizer function.')
parser.add_argument('--loss',
                    dest='loss',
                    default='categorical_crossentropy',
                    help='Name of tf.keras.loss function.')
parser.add_argument('--metrics',
                    dest='metrics',
                    default='accuracy',
                    help='Name of tf.keras.metrics function.')
args = parser.parse_args()

activation_options = ['elu', 'hard_sigmoid', 'linear', 'relu', 'selu',
                     'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh']

optimizer_options = ['adadelta', 'adagrad', 'adam', 'adamax', 'nadam', 'rmsprop', 'sgd']

loss_options = ['binary_crossentropy', 'categorical_hinge', 'cosine', 'cosine_proximity',
                'hinge', 'kullback_leibler_divergence', 'logcosh', 'mae', 'mape',
                'mean_absolute_error', 'mean_absolute_percentage_error', 'categorical_crossentropy',
                'mean_squared_logarithmic_error', 'mse', 'poisson', 'sparse_categorical_crossentropy',
                'squared_hinge']                

metrics_options = ['accuracy', 'auc', 'average_precision_at_k', 'false_negatives',
                   'false_negatives_at_thresholds', 'false_positives', 'false_positives_at_thresholds',
                   'mean', 'mean_absolute_error', 'mean_cosine_distance', 'mean_iou', 'mean_per_class_accuracy',
                   'mean_relative_error', 'mean_squared_error', 'mean_tensor', 'percentage_below',
                   'precision', 'precision_at_k', 'precision_at_thresholds', 'precision_at_top_k',
                   'recall', 'recall_at_k', 'recall_at_thresholds', 'recall_at_top_k', 'root_mean_squared_error',
                   'sensitivity_at_specificity', 'sparse_average_precision_at_k', 'sparse_precision_at_k',
                   'specificity_at_sensitivity', 'true_negatives', 'true_negatives_at_thresholds',
                   'true_positives', 'true_positives_at_thresholds']

assert(args.activation in activation_options)
assert(args.optimizer in optimizer_options)
assert(args.loss in loss_options)

valid_metrics = list()
for m in args.metrics.split(','):
    if m in metrics_options:
        valid_metrics.append(m)
    else:        
        logging.warn('%s is not a valid metric.  Ignoring.' % m)

tracks = pickle.load(open(os.path.join(args.data_path,'pev_starting_tracks.pkl')))
cascades = pickle.load(open(os.path.join(args.data_path,'pev_cascades.pkl')))

labeled_data = [(0,h) for h in cascades] + [(1,h) for h in tracks]
random.shuffle(labeled_data)

n_output_units = 2

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

input_layer = Input(shape=(28, 28, 1))
convolution_stage = Conv2D(32,
                           (2, 2), 
                           input_shape=(28, 28, 1),
                           activation=args.activation)(input_layer)
pooling_stage = MaxPooling2D(pool_size=(2,2))(convolution_stage) 
flatten_stage = Flatten()(pooling_stage)
output_layer = Dense(n_output_units, activation='softmax')(flatten_stage)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=args.optimizer,
              loss=args.loss,
              metrics=valid_metrics)

x_train = np.array([h for l,h in labeled_data[:200]])
x_train = x_train.reshape(200, 28, 28, 1)

y_train = np.array([l for l,h in labeled_data[:200]])
y_train_binary = to_categorical(y_train)
assert(len(x_train) == len(y_train))
model.fit(x_train, y_train_binary, epochs=args.epochs)

x_test = np.array([h for l,h in labeled_data[200:]])
print(x_test.shape)
x_test = x_test.reshape(200, 28, 28, 1)
y_test = np.array([l for l,h in labeled_data[200:]])
y_test_binary = to_categorical(y_test)
model.evaluate(x_test, y_test_binary)

model.summary()

#!/usr/bin/env python

import argparse
import logging

parser = argparse.ArgumentParser(description='Hello IceCube')
parser.add_argument('--n-hidden-units',
                    dest='n_hidden_units',
                    default=128,
                    help='Number of units in the hidden layer.')
parser.add_argument('--activation',
                    dest='activation',
                    default='relu',
                    help='Name of tf.keras.activation function.')
parser.add_argument('--optimizer',
                    dest='optimizer',
                    default='adam',
                    help='Name of tf.keras.optimizer function.')
parser.add_argument('--loss',
                    dest='loss',
                    default='log_loss',
                    help='Name of tf.keras.loss function.')
parser.add_argument('--metrics',
                    dest='metrics',
                    default='metrics',
                    help='Name of tf.keras.metrics function.')
args = parser.parse_args()

activtion_options = ['elu', 'hard_sigmoid', 'linear', 'relu', 'selu',
                     'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh']

optimizer_options = ['adadelta', 'adagrad', 'adam', 'adamax', 'nadam', 'rmsprop', 'sgd']

loss_options = ['absolute_difference', 'cosine_distance', 'hinge_loss',
                'huber_loss', 'log_loss', 'mean_pairwise_squared_error',
                'mean_squared_error', 'sigmoid_cross_entropy',
                'softmax_cross_entropy', 'sparse_softmax_cross_entropy']

metrics_options = ['accuracy', 'auc', 'average_precision_at_k', 'false_negatives',
                   'false_negatives_at_thresholds', 'false_positives', 'false_positives_at_thresholds',
                   'mean', 'mean_absolute_error', 'mean_cosine_distance', 'mean_iou', 'mean_per_class_accuracy',
                   'mean_relative_error', 'mean_squared_error', 'mean_tensor', 'percentage_below',
                   'precision', 'precision_at_k', 'precision_at_thresholds', 'precision_at_top_k',
                   'recall', 'recall_at_k', 'recall_at_thresholds', 'recall_at_top_k', 'root_mean_squared_error',
                   'sensitivity_at_specificity', 'sparse_average_precision_at_k', 'sparse_precision_at_k',
                   'specificity_at_sensitivity', 'true_negatives', 'true_negatives_ath_thresholds',
                   'true_positives', 'true_positives_at_thresholds']

assert(args.activation in activation_options)
assert(args.optimizer in optimizer_options)
assert(args.loss in loss_options)

valid_metrics = list()
for m in args.metrics.split(','):
    if m in metrics_options:
        valid_metrics.append(m)
    else:        
        logging.warn('%s is not a valid metric.  Ignoring.')

import tensorflow
import pickle
import numpy

data = pickle.load(open('bootcamp.pkl'))

training_data = data['training_data']
test_data = data['test_data']

n_output_units = len(set(training_data['labels']))

# Q1: How do I choose the number of units of the hidden layer?
# Q2: What's the best activation function to use?
hidden_layer = tf.keras.layers.Dense(args.n_hidden_units, activation=args.activation)
output_layer = tf.keras.layers.Dense(3)

# Start with single
network = [hidden_layer, output_layer]
model = tf.keras.models.Sequential(network)

model.compile(optimizer=args.optimizer,
              loss=args.loss,
              metrics=valid_metrics)

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)




        


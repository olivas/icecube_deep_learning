#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Hello IceCube')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--n-hidden-units',
                    dest='n_hidden_units',
                    default=128,
                    help='Number of units in the hidden layer.')
args = parser.parse_args()

import tensorflow
import pickle
import numpy

data = pickle.load(open('bootcamp.pkl'))

training_data = data['training_data']
test_data = data['test_data']

# Q1: How do I choose the number of units of the hidden layer?
#input_layer = tf.keras.layers.Flatten(input_shape=(28, 28))
#hidden_layer = tf.keras.layers.Dense(128, activation='relu')
#output_layer = tf.keras.layers.Dense(3, activation='softmax')

# start with linear (no activation)
hidden_layer = tf.keras.layers.Dense(args.n_hidden_units)
output_layer = tf.keras.layers.Dense(3)

# start with MLP
network = [hidden_layer, output_layer]

model = tf.keras.models.Sequential(network)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)




        


#!/usr/bin/env python

import argparse
import pickle
import numpy
import pylab

parser = argparse.ArgumentParser(description='Hello IceCube')
parser.add_argument('pickle_fn', help='Name of input pickle file.')
args = parser.parse_args()

# data is just a list of 2d numpy arrays.
data = pickle.load(open(args.pickle_fn))

for image in data:
    pylab.imshow(image)
    pylab.show()

import random
import copy
import numpy
from math import sqrt

def generate(histogram, function = lambda x, y : y): 
    '''
    Apply a shaping function to the histogram.
    '''
    result = copy.deepcopy(histogram)
    for idx, value in enumerate(histogram['bin_values']): 
        mean = function(idx, value)
        if mean > 0:
            sigma = sqrt(mean)
            result['bin_values'][idx] = int(random.gauss(mean, sigma))
    del result['_id']
    return result

def generate_data(histogram,
                  transformations,
                  label_mapping, 
                  n_samples = 1000):
    '''
    Returns a tuple (histograms, labels).
    We can use this both for training data and testing data.
    '''
    labels = list()
    histograms = list()
    for i in range(n_samples):
        label = random.choice(transformations.keys())
        function = transformations[label]
        labels.append(label_mapping[label])
        histograms.append(generate(histogram, function))
    return histograms, labels





        


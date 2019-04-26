#!/usr/bin/env python

import os
import argparse
import logging
import pickle
import numpy
import pylab

from icecube import icetray
from icecube import dataio
from icecube import dataclasses

parser = argparse.ArgumentParser(description='Hello IceCube')
parser.add_argument('gcdfilename', help='Path to GCDFile')
parser.add_argument('i3filename', help='Path to I3File.')
parser.add_argument('pickle_fn', help='Name of output pickle file.')
args = parser.parse_args()

def extract_inice_geometry(gcdfile):
    for frame in gcdfile:
        if "I3Geometry" in frame:
            return frame['I3Geometry'].omgeo

    
gcdfile = dataio.I3File(args.gcdfilename)
i3file = dataio.I3File(args.i3filename)

def whiten_histogram(histogram):
    max_bin = 0
    for row in histogram:
        if row.max() > max_bin:
            max_bin = row.max()
    return histogram/max_bin            

geometry = extract_inice_geometry(gcdfile)


histograms = list()
for frame in i3file:
    if frame.Stop == icetray.I3Frame.DAQ:
        if 'InIcePulses' in frame:

            x = list()
            y = list()
            charge = list()
            
            reco_pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'InIcePulses')
            for omkey, reco_pulse_series in reco_pulse_map.items():
                position = geometry[omkey].position
                x.append(position.x)
                y.append(position.z) # we want a side view.  that's not a typo.
                charge.append(sum([p.charge for p in reco_pulse_series]))

            H, xedges, yedges = numpy.histogram2d(x, y, weights = charge, bins = [28, 28])            
            histograms.append(whiten_histogram(H))
            
# dump a list of numpy 2d arrays 
pickle.dump(histograms, open(args.pickle_fn, 'w'))


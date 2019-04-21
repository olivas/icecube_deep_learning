#!/usr/bin/env python

from db import create_client
from histogram_transformations import generate
from histogram_transformations import generate_data

def get_histogram(database_name, collection_name, histogram_name):
    
    client = create_client()
    db = client[database_name]
    print("The following collections were found in %s" % database_name)
    for coll_name in db.collection_names():
        print("  %s" % coll_name)

    collection = db[collection_name]
    return collection.find_one({'name' : histogram_name})

def plot_histogram(histogram):

    import pylab
    
    bin_values = histogram['bin_values']
    nbins = len(bin_values)
    xmax = float(histogram['xmax'])
    xmin = float(histogram['xmin'])
    bin_width = (xmax - xmin)/float(nbins)
    x = [xmin + i*bin_width for i in range(nbins)]
    pylab.bar(x, [int(bv) for bv in bin_values])
    pylab.title(collection_name)
    pylab.xlabel(histogram_name)
    pylab.show()    
    
if __name__ == '__main__':

    import pickle
    
    database_name = 'simprod_histograms'
    collection_names = ['IceCube:2016:filtered:level2:CORSIKA-in-ice:20699',
                        'IceCube:2016:filtered:level2:CORSIKA-in-ice:20772',
                        'IceCube:2016:filtered:level2:CORSIKA-in-ice:20734',
                        'IceCube:2016:filtered:level2:CORSIKA-in-ice:20764'] 
    histogram_name = 'PrimaryEnergy'

    same = lambda x,y: y
    half = lambda x,y : 0.5 * x
    linear = lambda x,y : y + 0.1*x

    primary_energy = get_histogram(database_name, collection_names[0], 'PrimaryEnergy')

    transformations = {'same': same,
                       'half': half,
                       'linear': linear}
    label_mapping = {l: v for v, l in enumerate(transformations.keys())}
    
    histograms, labels = generate_data(primary_energy, transformations, label_mapping)
    training_data = {'histograms': histograms, 'labels': labels}
    
    histograms, labels = generate_data(primary_energy, transformations, label_mapping)
    test_data = {'histograms': histograms, 'labels': labels}
    
    data = {'training_data' : training_data,
            'test_data' : test_data}

    pickle.dump(data, open('bootcamp.pkl', 'w'))
        
#    for collection_name in collection_names:
#        primary_energy = get_histogram(database_name, collection_name, histogram_name)
#                
#        fake_histogram = generate(primary_energy, linear)
#         
#        plot_histogram(primary_energy)
#        plot_histogram(fake_histogram)

        
        

    


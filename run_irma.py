from __future__ import print_function



import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from da import dA
import gzip
import pickle

from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image



learning_rate=0.1
training_epochs=15
dataset='file.pklz'
batch_size=20
output_folder='irma_plots'

new_path = os.path.join(os.path.split(__file__)[0], "data", dataset)

with gzip.open(new_path, 'rb') as f:
    train_set, test_set = pickle.load(f)

train_x, train_paths = train_set
train_set_x = theano.shared(numpy.asarray(train_x, dtype=theano.config.floatX), borrow=True)

test_x, test_paths = train_set
test_set_x = theano.shared(numpy.asarray(test_x, dtype=theano.config.floatX), borrow=True)

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0]

# start-snippet-2
# allocate symbolic variables for the data
index = T.lscalar()    # index to a [mini]batch
x = T.matrix('x')  # the data is presented as rasterized images
# end-snippet-2

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)

####################################
# BUILDING THE MODEL NO CORRUPTION #
####################################

rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = dA(
    numpy_rng=rng,
    theano_rng=theano_rng,
    input=x,
    n_visible=32 * 32,
    n_hidden=32 * 32 / 2
)

cost, updates = da.get_cost_updates(
    corruption_level=0.,
    learning_rate=learning_rate
)

train_da = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size]
    }
)

start_time = timeit.default_timer()

############
# TRAINING #
############

# go through training epochs
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for batch_index in range(n_train_batches):
        c.append(train_da(batch_index))

    print('Training epoch %d, cost ' % epoch, numpy.mean(c))

end_time = timeit.default_timer()

training_time = (end_time - start_time)

print(('The no corruption code for file ' +
       os.path.split(__file__)[1] +
       ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)
image = Image.fromarray(
    tile_raster_images(X=da.W.get_value(borrow=True).T,
                       img_shape=(32, 32), tile_shape=(10, 10),
                       tile_spacing=(1, 1)))
image.save('filters_corruption_0.png')


# latent_rep = da.get_latent_representation()
# print(latent_rep)
# print(latent_rep.tag.test_value)


def transform(X):
    z = da.get_prediction()
    predict_da = theano.function([x],z)
    return predict_da(X)

X_prime = transform(train_set_x)


# start-snippet-3
#####################################
# BUILDING THE MODEL CORRUPTION 30% #
#####################################

rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = dA(
    numpy_rng=rng,
    theano_rng=theano_rng,
    input=x,
    n_visible=32 * 32,
    n_hidden=32 * 32 / 2
)

cost, updates = da.get_cost_updates(
    corruption_level=0.3,
    learning_rate=learning_rate
)

train_da = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size]
    }
)

start_time = timeit.default_timer()

############
# TRAINING #
############

# go through training epochs
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for batch_index in range(n_train_batches):
        c.append(train_da(batch_index))

    print('Training epoch %d, cost ' % epoch, numpy.mean(c))

end_time = timeit.default_timer()

training_time = (end_time - start_time)

print(('The 30% corruption code for file ' +
       os.path.split(__file__)[1] +
       ' ran for %.2fm' % (training_time / 60.)), file=sys.stderr)
# end-snippet-3

# # start-snippet-4
image = Image.fromarray(tile_raster_images(
    X=da.W.get_value(borrow=True).T,
    img_shape=(32, 32), tile_shape=(10, 10),
    tile_spacing=(1, 1)))
image.save('filters_corruption_30.png')
# # end-snippet-4

os.chdir('../')

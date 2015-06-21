import os
from urllib import urlretrieve
import gzip
import pickle
import numpy
import theano
import theano.tensor as T

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'mnist.pkl.gz'

def pickle_load(f, encoding):
    return pickle.load(f)

def _load_data(url=DATA_URL, filename=DATA_FILENAME):
    """Load data from `url` and store the result in `filename`."""
    #print 'filename for the minist datatset:',filename
    if not os.path.exists(filename):
        print("Downloading MNIST dataset")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        return pickle_load(f, encoding='latin-1')


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_data():
    """Get data with labels, split into training, validation and test set."""
    data = _load_data()
    xs, ys = data[0]
    test_set = (xs[0:10000,:],ys[0:10000])
    #test_set = (xs[400:600,:],ys[400:600]) #check the model fitting.
    #valid_set = (xs[200:400,:],ys[200:400])
    #train_set = (xs[400:numpoint,:],ys[400:numpoint])
    valid_set = test_set
    train_set = (xs[10000:,:],ys[10000:])
    
    print 'shape of x:',train_set[0].shape
    
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y), \
            (test_set_x, test_set_y)

    
if __name__ == '__main__':
    tnset,vset,teset = load_data()
    tnx ,tny = tnset
    print 'shape of x:',tnx.shape[0]



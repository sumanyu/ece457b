__author__ = 'vps'

import cPickle, gzip, numpy

# Trying to understand how MNIST is stored

f = gzip.open('data/mnist.pkl.gz', 'rb')

# tuple of train, validation and test set
train_set, valid_set, test_set = cPickle.load(f)

f.close()

# train_set is a tuple, first element is 2-d np array, 2nd element is 1d np array of class
X, y = train_set


print(len(X[0]))

print(type(train_set))
print(len(train_set))

print(type(train_set[0]))
print(len(train_set[0]))

print(type(train_set[1]))
print(len(train_set[1]))

# try our images

f1 = gzip.open('data/file.pklz', 'rb')

# tuple of train, validation and test set
train, test = cPickle.load(f1)

f.close()

# train_set is a tuple, first element is 2-d np array, 2nd element is 1d np array of class
X_train, paths = train

print(len(X_train[0]))
print(len(X_train))

print("")
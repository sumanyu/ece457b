__author__ = 'vps'

from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report
import numpy as np

from sknn import ae, mlp

mnist = fetch_mldata('mnist-original')
X_train, X_test, y_train, y_test = train_test_split(
        (mnist.data / 255.0).astype(np.float32),
        mnist.target.astype(np.int32),
        test_size=1.0/7.0, random_state=1234)


# Initialize auto-encoder for unsupervised learning.
myae = ae.AutoEncoder(
            layers=[
                ae.Layer("Tanh", units=128),
                ae.Layer("Sigmoid", units=64)],
            learning_rate=0.002,
            n_iter=10)

# Layerwise pre-training using only the input data.
myae.fit(X_train)

# Initialize the multi-layer perceptron with same base layers.
mymlp = mlp.Classifier(
            layers=[
                mlp.Layer("Tanh", units=128),
                mlp.Layer("Sigmoid", units=64),
                mlp.Layer("Softmax")])

# Transfer the weights from the auto-encoder.
myae.transfer(mymlp)

# Now perform supervised-learning as usual.
mymlp.fit(X_train, y_test)


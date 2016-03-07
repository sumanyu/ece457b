__author__ = 'vps'

from sklearn.datasets import fetch_mldata
import os
from numpy import arange
import random
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB


clf = MultinomialNB()

custom_data_home = os.path.join(os.path.split(__file__)[0],"data")

mnist = fetch_mldata('MNIST original', data_home=custom_data_home)


X, y = mnist.data / 255., mnist.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Fit the model
clf.fit(X_train, y_train)

# Perform the predictions
y_predicted = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print "Accuracy = {} %".format(accuracy_score(y_test, y_predicted)*100)

from sklearn.metrics import classification_report
print "Classification Report \n {}".format(classification_report(y_test, y_predicted, labels=range(0,10)))
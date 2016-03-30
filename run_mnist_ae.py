import os

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from DenoisingAutoencoder import DenoisingAutoencoder


custom_data_home = os.path.join(os.path.split(__file__)[0], "data")
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)


X, y = mnist.data / 255., mnist.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

#do a loop through a series of hidden size, and epochs
epochs = [3,4,5,6,7,8,9,10]
hidden_size = [100, 200, 300, 400]

for num_epochs in epochs:
    for num_hidden in hidden_size:
        print "EPOCHS: %d" % num_epochs
        print "HIDDEN: %d" % num_hidden
        da = DenoisingAutoencoder(n_hidden=num_hidden, verbose=True, training_epochs=num_epochs)
        da.fit(X_train)
        
        X_train_latent = da.transform_latent_representation(X_train)
        X_test_latent = da.transform_latent_representation(X_test)
        
        clf = MultinomialNB()
        
        # Fit the model
        clf.fit(X_train_latent, y_train)
        
        # Perform the predictions
        y_predicted = clf.predict(X_test_latent)
        
        print "Accuracy = {} %".format(accuracy_score(y_test, y_predicted)*100)
        
        print "Classification Report \n {}".format(classification_report(y_test, y_predicted, labels=range(0,10)))
        

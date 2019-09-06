"""
Trains a simple neural network on MNIST classification, using my implementation.
"""

from data.fashion_MNIST import *
from helper.NN import *
from helper.activations import *
from helper.losses import *
import numpy as np
import warnings

np.seterr(all='warn')
warnings.filterwarnings('error')
def train_model(id):
    data = load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)
 
    batch_size = 1
    epochs = 30
    lr = 1
    r = 0.0000

    mse = MeanSquaredCost()

    classifier = SimpleNeuronalNetwork((784, 20, 10), sigmoid_activation, sigmoid_derivation, mse)

    for e in range(epochs):
        for b in range(x_train.shape[0] // batch_size):
            for s in range(batch_size):
                classifier.train_network(x_train[b * batch_size + s], y_train[b * batch_size + s])
            classifier.apply_changes(lr, r)

        accuracy = 0        

        for s in range(x_train.shape[0]):
            output = classifier.get_output(x_train[s, :])
            if np.argmax(output) == np.argmax(y_train[s, : ]):
                accuracy += 1            

        print(accuracy / x_train.shape[0], flush = True)

train_model(1)

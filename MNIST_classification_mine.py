"""
Trains a simple Extreme Learning Machine on MNIST classification, using my implementation.
"""

from data.fashion_MNIST import *
from ELM import ELM
from helper.activations import *
import numpy as np

def train_model():
    data = load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)
 
    num_neurons = 2000

    model = ELM(784, num_neurons, 10, sigmoid_activation)

    model.train(x_train, y_train)
    

    # evaluate training accuracy

    train_accuracy = 0

    for s in range(x_train.shape[0]):
        output = model.get_output(x_train[s, :])
        if np.argmax(output) == np.argmax(y_train[s, : ]):
            train_accuracy += 1            

    print(train_accuracy / x_train.shape[0], flush = True)

    # evaluate test accuracy

    test_accuracy = 0

    for s in range(x_test.shape[0]):
        output = model.get_output(x_test[s, :])
        if np.argmax(output) == np.argmax(y_test[s, : ]):
            test_accuracy += 1            

    print(test_accuracy / x_test.shape[0], flush = True)

train_model()

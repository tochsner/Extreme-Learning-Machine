"""
A simple implementation of a vanilla Extreme Learning Machine.
"""

import numpy as np
from sklearn import linear_model

class ELM:

    def __init__(self, input_size, num_neurons, output_size, activation_function):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.output_size = output_size

        self.activation_function = activation_function

        self.random_weights = np.zeros((input_size, num_neurons))
        self.random_bias = np.zeros((num_neurons))
        self.trained_weights = np.zeros((output_size))
        self.trained_bias = 0

    def train(self, x, y):
        self.random_weights = np.random.randn(self.input_size, self.num_neurons)
        self.random_bias = np.random.randn(self.num_neurons)
        
        intermediate_activations = np.matmul(x, self.random_weights)
        intermediate_activations = self.activation_function(intermediate_activations + self.random_bias)

        lin_reg = linear_model.LinearRegression()
        model = lin_reg.fit(intermediate_activations, y)

        self.trained_weights = model.coef_
        self.trained_bias = model.intercept_

    def get_output(self, x):
        intermediate_activations = np.matmul(x, self.random_weights)
        intermediate_activations = self.activation_function(intermediate_activations + self.random_bias)

        output = np.dot(self.trained_weights, intermediate_activations) + self.trained_bias

        return output

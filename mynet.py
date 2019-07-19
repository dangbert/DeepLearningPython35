"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.
"""

#### Libraries
# Standard library
import random
import pickle
import os

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        # creates a column vector for each layer (except the first)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # creates an np array for layers 0...num_layers-1
        #   for a given layer col 0 = vector of weights for connections
        #   from node 0 (in cur layer) to next layer's nodes
        self.weights = [np.random.rand(sizes[i+1], sizes[i])
                        for i in range(self.num_layers-1)]

    # pickle data in this class as a backup
    # https://stackoverflow.com/a/2842727
    def save(self, filename):
        if os.path.exists(filename):
          os.remove(filename)
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
        print("*** network saved to '" + filename + "' ***")

    # load data into this class from pickle file
    def load(self, filename):
        f = open(filename, 'rb')
        self.__dict__.update(pickle.load(f))
        f.close()
        #print("*** network loaded from '" + filename + "' ***")

    # returns the output layers activations for given input
    def evaluate(self, x):
        return self.getActivations(x)[:-1]

    # returns list of the activation at each layer for given input
    def getActivations(self, a):
        activations = [a]
        for b,w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w,a)+b)
            activations.append(a)
        return activations

    #def SGD(self, training_data, epochs, mini_batch_size, eta,
    #        test_data=None, start_epoch=1):

    #def update_mini_batch(self, mini_batch, eta):


    # do backpropogation
    # (currently just calculates errors at each node)
    def backprop(self, cur, expected):
        activations = self.getActivations(cur)
        print("\nactivations:")
        print(activations)

        nabla_b = [np.zeros(b.shape, dtype=float) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=float) for w in self.weights]

        # array to store the calculated error at each node as we backpropogate
        # errors[l][j] = dC/dzl_0
        errors = [np.zeros((n, 1), dtype=float) for n in self.sizes]

        # traverse layers backwards
        # TODO: consider adding a dummy layer of to the beginning of the weights and biases vectors
        #       so the indicies make sense and match up with the errors array, etc
        for l in range(len(self.sizes)-1, 0, -1):
            index = l-1 # notes: subtract 1 from layer number to index into weights and biases arrays
            #print("at layer " + str(l))
            # traverse nodes in layer l
            for j in range(self.sizes[l]):
                # calculate errors[l][j]
                al_j = activations[l][j]
                # if we're on the last layer
                if l == len(self.sizes)-1:
                    #errors[l][j] = 2*(al_j-expected[j]) * al_j*(1-al_j) # BP1
                    # TODO: use sigmoid_prime and cost_derivative directly here
                    errors[l][j] = (al_j-expected[j]) * al_j*(1-al_j) # BP1
                else:
                    # BP2
                    total = 0
                    for k in range(0, self.sizes[l+1]):
                        total += self.weights[index+1][k][j] * errors[l+1][k][0]
                    errors[l][j] = al_j*(1-al_j) + total

                # TODO: as test update the bias and weights corresponding to the error just calculated
                # (not sure if doing this right)
                #self.biases[index][j] += -0.25 * errors[l][j]
                nabla_b[index][j] = errors[l][j]
                for k in range(0, self.sizes[l-1]):
                    nabla_w[index][j] = errors[l][j]*activations[l-1][k]
                    #self.weights[index][j] += -0.25 * errors[l][j]*activations[l-1][k]
        return (nabla_b, nabla_w)

    # returns the vector of partial derivatives for
    # the cost with respect to output activations
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    # signmoid function
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # derivative of sigmoid function
    @staticmethod
    def sigmoid_prime(z):
        return sigmoid(z) * (1 - sigmoid(z))

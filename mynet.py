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
import sys

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
        # TODO: network still trains slower than orignal, why? (1 epoch should get close to 90% accuracy)
        self.weights = [np.random.randn(sizes[i+1], sizes[i])
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

    def SGD(self, training_data, epochs, mini_batch_size, rate,
            test_data=None, start_epoch=1):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        backup_dir = "backups"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        if start_epoch == 1:
            self.save(backup_dir + "/initial.pkl")

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            print("INITIAL: {} / {}".format(self.test(test_data), n_test));

        for n in range(start_epoch, epochs+1):
            sys.stdout.flush()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.updateMiniBatch(mini_batch, rate)
            if test_data:
                print("Epoch {} : {} / {}".format(n, self.test(test_data), n_test));
            else:
                print("Epoch {} complete".format(n))

            if n % 5 == 0:
                self.save(backup_dir + "/latest.pkl")
            if n % 25 == 0:
                self.save(backup_dir + "/epoch" + str(n) + ".pkl")

        self.save(backup_dir + "/epoch" + str(epochs) + ".pkl")
        print("\ntraining complete (reached epoch" + str(epochs) + ")")

    def test(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.getOutput(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # TODO: there's a way to more efficiently do this
    # (combining all data in the batch into a single array...)
    def updateMiniBatch(self, miniBatch, rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``rate``
        is the learning rate.
        note: to train on a single sample, set mini_batch=[(x,y)]
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # average the delta_b, delta_w calculated for each training sample
        # and adjust the weights, biases using the learning rate
        for x, y in miniBatch:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb+db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw+dw for nw, dw in zip(nabla_w, delta_w)]
        self.biases = [b-(rate/len(miniBatch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(rate/len(miniBatch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

    # do backpropogation
    # returns nabla_b, nabla_w
    # (partial dertivatives of biases and weights wrt. cost function)
    def backprop(self, cur, expected):
        # TODO: also use zs from this function?
        zs, activations = self.feedforward(cur)

        # note: nabla_b is the same as the "error" calculated at each node
        nabla_b = [np.zeros(b.shape, dtype=float) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=float) for w in self.weights]

        # array to store the calculated error at each node as we backpropogate
        errors = [np.zeros((n, 1), dtype=float) for n in self.sizes]
        # traverse layers backwards
        # TODO: consider adding a dummy layer of to the beginning of the weights and biases vectors
        #       so the indicies make sense and match up with the errors array, etc
        for l in range(len(self.sizes)-1, 0, -1):
            index = l-1 # notes: subtract 1 from layer number to index into weights and biases arrays
            # traverse nodes in layer l
            for j in range(self.sizes[l]):
                # calculate errors[l][j]
                al_j = activations[l][j]
                # if we're on the last layer
                if l == len(self.sizes)-1:  # this is the equivalent of me calculating delta
                    # TODO: use sigmoid_prime and cost_derivative directly here
                    errors[l][j] = (al_j-expected[j]) * al_j*(1-al_j) # BP1
                else:
                    # BP2
                    total = 0
                    for k in range(0, self.sizes[l+1]):
                        total += self.weights[index+1][k][j] * errors[l+1][k][0]
                    errors[l][j] = al_j*(1-al_j) * total

                nabla_b[index][j] = errors[l][j]
                for k in range(0, self.sizes[l-1]):
                    nabla_w[index][j][k] = errors[l][j]*activations[l-1][k]
        return (nabla_b, nabla_w)

    # returns the output layers activations for given input
    def getOutput(self, x):
        _, activations = self.feedforward(x)
        return activations[-1]

    # returns list of z values and list of activations at each layer for given input
    def feedforward(self, a):
        activations = [a]
        zs = [np.zeros(a.shape, dtype=float)]   # zs[0] filled with dummy values
        for b, w in zip(self.biases, self.weights):
            zs.append(np.dot(w, activations[-1]) + b)
            activations.append(self.sigmoid(zs[-1]))
        return zs, activations

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
        return sigmoid(z) * (1.0 - sigmoid(z))

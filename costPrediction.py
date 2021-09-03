r"""
costPrediction.py
~~~~~~~~~~

Modified version of mynet.py to test having the network also predict its own cost
TODO: use inheritance like mynetExperiment.py does...
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
        r"""The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.

        Args:
            sizes (:obj:`list` of :obj:`int`): list of layer sizes (e.g. `[2,3,2]`)

        Attributes:
            sizes (:obj:`list` of :obj:`int`): list of layer sizes (e.g. `[2,3,2]`)
            num_layers (int): number of layers in network
            biases (:obj:`list` of :obj:`numpy.ndarray`):
                stores a column vector for each layer's biases (except the input layer). Initialized randomly using np.random.randn (for a gaussian distribution)
                example for sizes `[2,3,2]`:
                $$[\begin{bmatrix} [-0.25]\\ [-0.34]\\ [0.21] \end{bmatrix}, \begin{bmatrix} [-0.88]\\ [0.47]\\ \end{bmatrix}]$$

            weights (:obj:`list` of :obj:`numpy.ndarray`):
                list of weights in the network (one np array for each layer)

                for a given layer, col 0 = vector of weights for connections
                from node 0 (in cur layer) to next layer's nodes
                ample for sizes `[2,3,2]`: $$[\begin{bmatrix} [-0.3, -0.01]\\ [2.94, -0.88]\\ [0.91, -1.85] \end{bmatrix}, \begin{bmatrix} [1.62, 1.21, 0.95]\\ [-0.19, -0.68, 0.61]\\ \end{bmatrix}]$$
                let $w=weights[0]$ (stores all weights coming into layer 1 from layer 0).

                $w[j]$ is the list of weights coming into node $j$ in layer 1,
                so $w_jk = w[j][k]$ is the weight between the $j$th neuron in layer 1, and the $k$th neuron in layer 0.

                This ordering allows us to calculate activations with: $a' = \sigma(wa+b)$
        """
        sizes[-1] = sizes[-1] + 1 # add extra output for predicting cost

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.epoch = 0 # meta-param (how many epochs of training we've undergone)

        self.weights = [np.random.randn(sizes[i+1], sizes[i])
                        for i in range(self.num_layers-1)]
        # TODO: network still trains slower than orignal, why? (1 epoch should get close to 90% accuracy)

    def save(self, filename):
        """
        pickle data in this class as a backup to desired filename
        https://stackoverflow.com/a/2842727
        """
        if os.path.exists(filename):
          os.remove(filename)
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
        print("*** network saved to '" + filename + "' ***")

    def load(self, filename):
        """
        load data from pickle file into this class to overwrite its data
        Incomplete: (returns a Network object)
        """
        f = open(filename, 'rb')
        self.__dict__.update(pickle.load(f))
        f.close()
        print("*** network loaded from '" + filename + "' ***")

    def test(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        print("first test data (with y = {}):".format(test_data[0][1]))
        output = self.getOutput(test_data[0][0])
        print('output = ')
        print(output)
        print("argmax = {}".format(np.argmax(output[:-1])))

        test_results = [(np.argmax(self.getOutput(x)[:-1]), y)
                        for (x, y) in test_data]
        numCorrect = sum(int(x == y) for (x, y) in test_results)

        outputs = [self.getOutput(x) for (x, _) in test_data]
        print("outputs[0] = ")
        print(outputs[0])
        print("test_data[0][1] = ")
        print(test_data[0][1])
        costInfo = [self.cost_derivative(output_activations, y) for (output_activations, (x, y)) in zip(outputs, test_data)]
        sumCostQ = 0 
        sumPq = 0
        for (_, costQ, pQ) in costInfo:
            sumCostQ += costQ
            sumPq += abs(pQ)
        return (numCorrect, sumCostQ, sumPq)

    def getOutput(self, x):
        """
        returns the output layers activations for given input (using feedforward())
        """
        _, activations = self.feedforward(x)
        return activations[-1]

    def feedforward(self, a):
        """
        returns list of z values and list of activations at each layer for given input
        """
        activations = [a]
        zs = [np.zeros(a.shape, dtype=float)]   # zs[0] filled with dummy values
        for b, w in zip(self.biases, self.weights):
            zs.append(np.dot(w, activations[-1]) + b)
            activations.append(self.sigmoid(zs[-1]))
        return zs, activations

    # TODO: there's a way to more efficiently do this!
    #   (combining all data in the batch into a single matrix, as the book mentions)
    #   (takes advantage of parallelization in CPU and GPU's)
    #   try to work this out myself but if needed reference:
    #   https://medium.com/@hindsellouk13/matrix-based-back-propagation-fe143ce2b2df
    def updateMiniBatch(self, miniBatch, rate):
        """
        Perform gradient descent using backpropogation on a single miniBatch
        and update the network's weights and biases.


        Args:
            miniBatch (:obj:`list` of :obj:`tuples`): List of training inputs / expected outputs (x,y).
                Calculated changes to the weights and biases will be averaged across this mini batch.

                Note: to train on a single sample, set mini_batch=[(x,y)]
            rate (:obj:`float`): Learning rate to use for training (e.g. 0.05).
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # average the delta_b, delta_w calculated for each training sample
        # and adjust the weights, biases using the learning rate
        count = -1 
        for x, y in miniBatch:
            count += 1
            delta_b, delta_w = self.backprop(x, y, count == len(miniBatch) - 1)
            nabla_b = [nb+db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw+dw for nw, dw in zip(nabla_w, delta_w)]
        self.biases = [b-(rate/len(miniBatch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(rate/len(miniBatch))*nw
                        for w, nw in zip(self.weights, nabla_w)]


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
        n_train = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            (n_correct, sumCostQ, sumPq) = self.test(test_data)
            print("INITIAL {}: {:.2f}% -- {} / {}\tsumCostQ = {:.4f}\tsumPq = {:.34f}".format(i, (n_correct/n_test*100), n_correct, n_test, sumCostQ, sumPq))

        for i in range(start_epoch, epochs+1):
            self.epoch += 1
            sys.stdout.flush()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n_train, mini_batch_size)]
            batchNum = 0
            for mini_batch in mini_batches:
                #print("at mini batch {} of {}".format(batchNum+1, len(mini_batches)))
                self.updateMiniBatch(mini_batch, rate)
                batchNum += 1
            if test_data:
                (n_correct, sumCostQ, sumPq) = self.test(test_data)
                print("Epoch {}: {:.2f}% -- {} / {}\tsumCostQ = {:.4f}\tsumPq = {:.34f}".format(i, (n_correct/n_test*100), n_correct, n_test, sumCostQ, sumPq))
            else:
                print("Epoch {} complete".format(i))

            if i % 10 == 0:
                self.save(backup_dir + "/latest.pkl")
            if i % 25 == 0:
                self.save(backup_dir + "/epoch" + str(i) + ".pkl")

        self.save(backup_dir + "/epoch" + str(epochs) + ".pkl")
        self.save(backup_dir + "/latest.pkl")
        print("\ntraining complete (reached epoch" + str(epochs) + ")")

    def backprop(self, x, y, printCost=False):
        """
        do backpropagation
        returns nabla_b, nabla_w (partial dertivatives of biases and weights wrt. cost function)
        """
        zs, activations = self.feedforward(x)
        # note: nabla_b is the same as the "error" calculated at each node
        nabla_b = [np.zeros(b.shape, dtype=float) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=float) for w in self.weights]
        # array to store the calculated error at each node as we backpropagate
        errors = [ np.zeros((n, 1), dtype=float) for n in self.sizes[1:] ]

        # using the index -l we will traverse the layers backwards
        #   (convenient for indexing into weights and biases arrays)
        for l in range(1, len(self.sizes)):
            # NOTE: using the matrix forms of the equations instead of looping over the nodes in the
            #   current layer is way faster (50x difference in one test)
            if l == 1: # BP1:
                (costDeriv, costQ, pQ) = self.cost_derivative(activations[-l], y)
                errors[-l] = costDeriv *  self.sigmoid_prime(zs[-l])
                #if printCost and random.randint(0, 1000) == 0:
                #    print("costQ = {}\t confidence difference = {}".format(costQ, pQ - costQ))
            else:      # BP2:
                # (np.multiply() does the hadmard product, np.dot() does matrix multiplication)
                errors[-l] = np.multiply( np.dot(self.weights[-l+1].transpose(),errors[-l+1]),  self.sigmoid_prime(zs[-l]) )
            nabla_b[-l] = errors[-l] # BP3
            nabla_w[-l] = np.dot(errors[-l], activations[-l-1].transpose()) # BP4 (a matrix form)
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y_initial):
        r"""
        Return the vector of partial derivatives $\partial C_x / \partial a$
        for the output activations.
        (For the quadratic cost function $C$ in particular)

        Args:
            output_activations: vector of output activations
            y_initial: expected output vector (before the quadratic cost is appended)
        """
        # note: y will have a length one shorter than output_activations
        #   we will resize y, its last element will have value costQ (actual quadratic cost)
        print("y_initial = ")
        print(y_initial)
        print("y_initial shape = ")
        print(y_initial.shape)
        print("output_activations shape = ")
        print(output_activations.shape)
        y = np.copy(y_initial)
        y.resize((y.shape[0] + 1, 1))
        costQ = self.quadratic_cost(output_activations, y)
        y[-1] = costQ
        costFinal = (output_activations - y)
        pQ = output_activations[-1] # predicted costQ
        costFinal[-1] = costQ * (pQ - costQ) # overwrite last element in vector
        return (costFinal, costQ, pQ)

    def quadratic_cost(self, output_activations, y):
        r"""
        Returns the quadratic cost for the output activations.
        Args:
            output_activations: vector of output activations
            y: expected output vector
        """
        tmp = output_activations[:-1] - y[:-1]
        return np.dot(tmp[0], tmp[0]) / 2

    # signmoid function
    #@staticmethod
    def sigmoid(self, z):
        """
        Implements the sigmoid function.
        https://en.wikipedia.org/wiki/Sigmoid_function
        """
        return 1.0 / (1.0 + np.exp(-z))

    # derivative of sigmoid function
    #@staticmethod
    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

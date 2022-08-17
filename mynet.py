r"""
mynet.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.

More info: http://neuralnetworksanddeeplearning.com/chap1.html
http://neuralnetworksanddeeplearning.com/chap2.html
"""

#### Libraries
# Standard library
import random
import pickle
import os
import sys

# Third-party libraries
import numpy as np
from datetime import datetime

class Network(object):

    def __init__(self, sizes, backupDir="backups", name=datetime.now().strftime('%Y_%m_%d_%H_%M_%S')):
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
            name (str): optional, used for setting filepath for saved weights.

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
                so $w_{jk} = w[j][k]$ is the weight between the $j$th neuron in layer 1, and the $k$th neuron in layer 0.

                This ordering allows us to calculate activations with: $a' = \sigma(wa+b)$
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.epoch = 0 # meta-param (how many epochs of training we've undergone)

        self.weights = [np.random.randn(sizes[i+1], sizes[i])
                        for i in range(self.num_layers-1)]

        self.name = name
        self.backupDir = backupDir
        #self.backupDir = "{}/{}".format(backupDir, self.name)

    def save(self, filename):
        """
        pickle data in this class as a backup to desired filename
        https://stackoverflow.com/a/2842727

        Args:
            filename (str): name fo file to save (relative to `self.backupDir/self.name`)
        """
        saveDir = os.path.join(self.backupDir, self.name)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        filename = os.path.join(saveDir, filename)
        if os.path.exists(filename):
          os.remove(filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)
        print("*** network saved to '" + filename + "' ***")

    def load(self, fname):
        """
        load data from pickle file into this class to overwrite its data.
    
        Args:
            fname (str): name of pickle file to open.
                This path is first tried relative to self.backupDir, then its tried by iteself as a fallback.
        """
        saveDir = os.path.join(self.backupDir, self.name)

        relName = os.path.join(saveDir, fname)
        fname = relName if os.path.exists(relName) else fname
        with open(fname, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        print("*** network loaded from '{}' at epoch {} ***".format(fname, self.epoch))

    def test(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.getOutput(x)), y)
                        for (x, y) in test_data]
        # count the instances where the most activate output neuron matches the expected output index
        return sum(int(x == y) for (x, y) in test_results)

    # TODO: have this also return the number correct (like test() does above to replace it)
    def testQuadraticCost(self, test_data):
        """Return the quadratic cost of the provided test_data.
        Where test_data is a list of 2-tuples, the first element
        being an input vector to the network, and the second as
        the ground-truth output vector."""
        diffs = [self.getOutput(x) - y for (x, y) in test_data] # list of differences between output vectors
        return 1 / (2 * len(test_data)) * sum([np.sum(d * d) for d in diffs]) # 1/(2n) * (sum squared errors)

    def getOutput(self, x):
        """
        returns the output layers activations for given input (using feedforward())
        """
        _, activations = self.feedforward(x)
        return activations[-1]

    def feedforward(self, a):
        """
        returns list of z values and list of activations at each layer for a given input.
        (the first layer's "activations" will be identical to the inputs provided to this function).
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
        for x, y in miniBatch:
            delta_b, delta_w, _ = self.backprop(x, y)
            nabla_b = [nb+db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw+dw for nw, dw in zip(nabla_w, delta_w)]
        self.biases = [b-(rate/len(miniBatch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(rate/len(miniBatch))*nw
                        for w, nw in zip(self.weights, nabla_w)]


    def SGD(self, training_data, epochs, mini_batch_size, rate,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs (a vector) and the desired
        outputs (as an integer index, OR as a vector).  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        
        Args:
            epochs (int): epoch number to stop training at (given that we're starting at epoch self.epoch).
        
        """

        #start_epoch = start_epoch if start_epoch != None else self.epoch # optional param start_epoch
        if self.epoch == 0:
            self.save("initial.pkl")

        training_data = list(training_data)
        n_train = len(training_data)

        test_data = list(test_data) if test_data else None

        def getCost():
            """wrapper function (for now) for both ways of evaluating cost"""
            cost, n_test, n_correct = -1, len(test_data), -1
            if isinstance(test_data[0][1], np.ndarray):
                # test data was provided as output vectors (rather than as a simple index):
                cost = self.testQuadraticCost(test_data)
            else:
                n_correct = self.test(test_data)
            return cost, n_test, n_correct

        if test_data:
            cost, n_test, n_correct = getCost()
            print("INITIAL COST: {:.4f}\tcorrect: {:.2f}% -- {} / {}".format(cost, (n_correct/n_test)*100, n_correct, n_test))

        for _ in range(epochs - self.epoch):
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
                cost, n_test, n_correct = getCost()
                print("Epoch {}: COST: {:.4f}\tcorrect: {:.2f}% -- {} / {}".format(self.epoch, cost, (n_correct/n_test)*100, n_correct, n_test))
            else:
                print("Epoch {} complete".format(self.epoch))

            if self.epoch % 10 == 0:
                self.save("latest.pkl")
            if self.epoch % 25 == 0:
                self.save("epoch{}.pkl".format(str(self.epoch).rjust(4, '0'))) # 0 pad epoch

        self.save("epoch{}.pkl".format(str(self.epoch).rjust(4, '0')))
        self.save("latest.pkl")
        print("\ntraining complete (reached epoch {})".format(epochs))

    def backprop(self, x, y):
        """
        does backpropagation and returns returns nabla_b, nabla_w, nabla_a0
        (partial derivatives of biases and weights wrt. cost function, 
        nabla_a0 is dC/da for the activations ("inputs") of the input layer).
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
                errors[-l] = self.cost_derivative(activations[-l], y) * self.sigmoid_prime(zs[-l])
            else:      # BP2:
                # (np.multiply() does the hadmard product, np.dot() does matrix multiplication)
                errors[-l] = np.multiply( np.dot(self.weights[-l+1].transpose(),errors[-l+1]),  self.sigmoid_prime(zs[-l]) )
            nabla_b[-l] = errors[-l] # BP3
            nabla_w[-l] = np.dot(errors[-l], activations[-l-1].transpose()) # BP4 (a matrix form)

        nabla_a0 = self.weights[0].transpose() @ errors[0] # @ is another way to do matrix multiplication (same as np.dot() in this case)
        return (nabla_b, nabla_w, nabla_a0)

    def cost_derivative(self, output_activations, y):
        r"""
        Return the vector of partial derivatives $\partial C_x / \partial a$
        for the output activations.
        (For the quadratic cost function $C$ in particular)

        Args:
            output_activations: vector of output activations
            y: expected output vector
        """
        return (output_activations-y)

    # signmoid function
    #@staticmethod
    def sigmoid(self, z):
        """
        Implements the sigmoid function.
        https://en.wikipedia.org/wiki/Sigmoid_function
        """
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_inverse(self, z):
        """
        Implements the inverse of the sigmoid function.
        https://en.wikipedia.org/wiki/Logit
        """
        return np.log(z / (1 - z))

    # derivative of sigmoid function
    #@staticmethod
    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

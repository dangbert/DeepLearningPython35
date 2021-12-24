r"""
GrammarNet.py
~~~~~~~~~~

inherits from mynet.py
"""

import os, sys
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

import mynet
from datetime import datetime
import sys
import random

# Third-party libraries
import numpy as np

class Network(mynet.Network):
    """overload mynet.Network class"""
    def __init__(self, sizes, backupDir="backups", name=datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), grammarLayer=-1, otherNets=[]):
        # call parent constructor
        super().__init__(sizes, backupDir, name)

        # additional data members:
        self.grammarLayer = grammarLayer
        self.otherNets = otherNets

    def SGD(self, training_data, epochs, mini_batch_size, rate,
            test_data=None):
        """
        exact copy of mynet.py:SGD()
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


    def updateMiniBatch(self, miniBatch, rate):
        """
        Perform gradient descent using backpropogation on a single miniBatch
        and update the network's weights and biases.


        Args:
            miniBatch (:obj:`list` of :obj:`tuples`): List of training inputs / expected outputs (x,y).
                Calculated changes to the weights and biases will be averaged across this mini batch.

                Note: to train on a single sample, set mini_batch=[(x,y)]
            rate (:obj:`float`): Learning rate to use for training (e.g. 0.05).

            otherNets (array of :obj:`Network`): other networks that utilize the given grammar layer index
                these nets should start with a copy of the layer in this network prior to this grammar layer.
                b/c its easier to compute the error of the grammar layer in the other net this way (using backprop()).
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # average the delta_b, delta_w calculated for each training sample
        # and adjust the weights, biases using the learning rate
        for x, y in miniBatch:
            # errorAdj starts as an array of zeros for every layer (omitting the input layer)
            errorAdj = [np.zeros((n, 1), dtype=float) for n in self.sizes[1:]]
            # get activations of grammar layer
            gInput = x if self.grammarLayer == -1 else self.feedforward(x)[1][self.grammarLayer]

            for oNet in self.otherNets:
                # nabla_b (errors) of layer 1 (should correspond to grammar layer)
                #res = oNet.backprop(gInput, y, None) # must explicitly pass None so super().backprop() is not called instead...
                _, _, nabla_a0 = oNet.backprop(gInput, y)
                #import pdb; pdb.set_trace()
                # compute blame relative to z0 (if inputs a0 = sigmoid(z0) for some z0)
                nabla_z0 = np.multiply(self.sigmoid_prime(self.sigmoid_inverse(gInput)), nabla_a0) # (gInput is a0)

                errorAdj[self.grammarLayer - 1] += nabla_z0  # subtract 1 from index cause errorAdj omits the input layer
            #print('errorAdj:'); print(errorAdj)

            #print('flag1, x.shape = {}, y.shape = {}, len(errorAdj) = {}'.format(x.shape, y.shape, len(errorAdj)))
            #import pdb; pdb.set_trace()
            delta_b, delta_w, _ = self.backprop(x, y, errorAdj)
            nabla_b = [nb+db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw+dw for nw, dw in zip(nabla_w, delta_w)]
        self.biases = [b-(rate/len(miniBatch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(rate/len(miniBatch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

    # TODO: pass net0 as a param here?
    def backprop(self, x, y, errorAdj):
        """
        does backpropagation and returns returns nabla_b, nabla_w, nabla_a0
        (partial dertivatives of biases and weights wrt. cost function, 
        nabla_a0 is dC/da for the activations ("inputs") of the input layer).

        Args:
            errorAdj (array of numpy vectors): optional, additional values to add to errors while computing it.
                specifically this is used to adjust the error in a "Grammar" layer that also connects to another subnetwork.

        TODO: add support for computing dC/da for the activations ("inputs") of the input layer...
        """
        zs, activations = self.feedforward(x)
        # note: nabla_b is the same as the "error" calculated at each node
        nabla_b = [np.zeros(b.shape, dtype=float) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=float) for w in self.weights]
        # array to store the calculated error at each node as we backpropagate
        errors = errorAdj if errorAdj else [ np.zeros((n, 1), dtype=float) for n in self.sizes[1:] ]

        # using the index -l we will traverse the layers backwards
        #   (convenient for indexing into weights and biases arrays)
        for l in range(1, len(self.sizes)):
            # NOTE: using the matrix forms of the equations instead of looping over the nodes in the
            #   current layer is way faster (50x difference in one test)
            if l == 1: # BP1:
                errors[-l] += self.cost_derivative(activations[-l], y) * self.sigmoid_prime(zs[-l])
            else:      # BP2:
                # (np.multiply() does the hadmard product, np.dot() does matrix multiplication)
                errors[-l] += np.multiply( np.dot(self.weights[-l+1].transpose(),errors[-l+1]),  self.sigmoid_prime(zs[-l]) )
            nabla_b[-l] = errors[-l] # BP3
            nabla_w[-l] = np.dot(errors[-l], activations[-l-1].transpose()) # BP4 (a matrix form)

        #a0 = np.dot(self.weights[-l+1].transpose(),errors[-l+1]),  self.sigmoid_prime(zs[-l]) )
        # compute the change in cost relative to the inputs ("activations" of layer 0)
        a0 = self.weights[0].transpose() @ errors[0] # @ is another way to do matrix multiplication (same as np.dot() in this case)

        return (nabla_b, nabla_w, a0)
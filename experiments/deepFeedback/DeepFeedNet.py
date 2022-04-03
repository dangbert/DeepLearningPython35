r"""
DeepFeedNet.py
~~~~~~~~~~

Inherits from mynet.py, adjusting it as needed for this experiment.
"""

import os, sys
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

import mynet
from datetime import datetime
import sys
import random
import copy

# Third-party libraries
import numpy as np

class Network(mynet.Network):
    """
    Overload mynet.Network class to create a network that passes outputs from a deep layer back to its input layer.
    So for a given "raw" input, the network can be ran for several iterations, and hopefully improve its performance over the iterations.
    (On the first iteration, we inject random noise (or all zeros?) as a placeholder for the initial deep feedback).
    

    adjusts provided sizes as needed so second to last layer can be sent as feedback to first later
    (augmenting the "raw" input to the network by expanding its dimensionality).

    """
    def __init__(self, sizes, backupDir="backups", name=datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), grammarLayer=-1, otherNets=[]):
        sizes = copy.deepcopy(sizes)

        self.feedbackLayer = -2 # index of layer to capture output from as feedback for input layer
        self.feedbackDim = sizes[self.feedbackLayer] # dimensions of feedback layer
        sizes[0] += self.feedbackDim

        # call parent constructor
        super().__init__(sizes, backupDir, name)

    def test(self, test_data):
        """exact copy of mynet's function"""
        test_results = [(np.argmax(self.getOutput(x)), y)
                        for (x, y) in test_data]
        # count the instances where the most activate output neuron matches the expected output index
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data, epochs, mini_batch_size, rate,
            test_data=None, feedforward=None):
      """thin wrapper of mynet.SGD, (tells it to use the feedforward function defined in this class instead)."""
      return super().SGD(training_data, epochs, mini_batch_size, rate, feedforward=self.feedforward)
      # TODO: could we instead do something like:
      #   super().feedforward = self.feedforward()?

    #def getOutput(self, x, iter=0, totalIter=2, prevFeedback=None):
    def getOutput(self, x):
        """
        identical to mynet's version, but enssures self.feedforward is called instead of the parent one
        """

        _, activations = self.feedforward(x)
        return activations[-1]

        # TODO: if we need to implement an exact copy of any parent functions, can we just call super().getOutput(self.feedforward)
        #   i.e. make the parent take an optional param housing the function to call for a key operation...

        """
        #print(f"in DeepFeedNet.getOutput(), iter={iter+1}/{totalIter}")
        rawInputDim = len(x) # e.g. 784
        augX = copy.deepcopy(x)
        #augX.resize((self.sizes[0], 1)) # inserts zeros as new entries at end
        #np.testing.assert_array_equal(x, augX[:len(x)])

        if prevFeedback is None:
          prevFeedback = np.zeros((self.feedbackDim, 1))

        # append prevFeedback to raw input x
        augX = np.append(augX, prevFeedback, axis=0)
        # fill end of augX with prevFeedback
        #import pdb; pdb.set_trace()
        #augX[rawInputDim: ] = prevFeedback
        np.testing.assert_array_equal(augX[rawInputDim: ], prevFeedback)

        _, activations = self.feedforward(augX)

        # TODO: now recursively call self.getOutput as needed! :)
        if iter+1 < totalIter:
          # more iterations needed
          return self.getOutput(x, iter+1, totalIter, prevFeedback=activations[self.feedbackLayer])


        #import pdb; pdb.set

        return activations[-1]
        """

    def feedforward(self, rawA, iter=0, totalIter=2, prevFeedback=None):
        """
        returns list of z values and list of activations at each layer for a given input.
        (the first layer's "activations" will be identical to the inputs provided to this function).

        expands raw input (rawA) as needed based on iteration
        and returns final activations (recursively).

        params:
          iter: current iteration number (starts at 0)
          totalIter: number of iterations (of deep feedback) to perform.

        returns:
          activations of each layer of the network in the final iteration.
        """

        #print(f"in DeepFeedNet.feedforward(), iter={iter+1}/{totalIter}")
        rawInputDim = len(rawA) # e.g. 784
        if prevFeedback is None:
          prevFeedback = np.zeros((self.feedbackDim, 1))

        # append prevFeedback to raw input a
        augA = copy.deepcopy(rawA)
        augA = np.append(augA, prevFeedback, axis=0)
        np.testing.assert_array_equal(augA[rawInputDim: ], prevFeedback)

        activations = [augA]
        zs = [np.zeros(augA.shape, dtype=float)]   # zs[0] filled with dummy values
        for b, w in zip(self.biases, self.weights):
            zs.append(np.dot(w, activations[-1]) + b)
            activations.append(self.sigmoid(zs[-1]))

        # recursively call self.feedforward as needed:
        if iter+1 < totalIter:
          return self.feedforward(rawA, iter+1, totalIter, prevFeedback=activations[self.feedbackLayer])
        return zs, activations

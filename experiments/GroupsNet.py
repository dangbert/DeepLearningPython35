r"""
GroupsNet.py
~~~~~~~~~~

inherits from mynet.py, but tests the possible allowing a layer l to be partitioned into g groups.
   The following layer (l+1) is also paritioned into g groups, and 
   only nodes within the same corresponding groups will have connections between them across the two layers.
   e.g. the first 10 nodes of layer 1, will only have connections to the first 7 nodes of layer 2.
   we can accomplish this easily by zeroing out (permanently) the majority of the weights between the layers, and zeroing out nabla_w entries
   for these zero'd weights (see self.masks).
"""

import mynet

# Third-party libraries
import numpy as np

class Network(mynet.Network):
    """overload mynet.Network class"""
    def __init__(self, sizes):
        # call parent constructor
        super().__init__(sizes)

        # harcoded for sizes [784, 100, 70, 30, 10] for now:
        #           groups:  [1,    10, 10,  1,  1] (make this another input)
        # create masks

        self.groups = [1, 10, 1, 1, 1] # to split later with 100 nodes into 10 groups


        # mask (all 1s and 0s) to apply to self.weights (zeroing out certain weights)
        weightsMask = [np.ones(w.shape) for w in self.weights]

        # weights between layers 2 and 1 (where layer 0 is input layer):
        #   (70x100 matrix)

        l = 1 # layer index (that has more than 1 group...)
        self.special = l # store for use later below...
        weightsMask[l] = np.zeros(weightsMask[l].shape)

        numGroups = self.groups[l] # e.g. 10
        if sizes[l] % numGroups != 0:
            print("ERROR: layer {} has size {}, not divisible by numGroups {}".format(l, sizes[l], numGroups))
        groupSize = int(sizes[l] / numGroups) # size of groups in layer l e.g. 10 (7 groups of size 10, for 70 nodes total)
        if sizes[l+1] % numGroups != 0:
            print("ERROR: layer {} has size {}, not divisible by numGroups {}".format(l, sizes[l], numGroups))
        nextGroupSize = int(sizes[l+1] / numGroups) # size of groups in layer l+1
        print('groupSize = {}'.format(groupSize))
        print('setting up mask for weights[{}] with shape {}'.format(l, weightsMask[1].shape))
        print('numGroups = {}'.format(numGroups))

        # for every node in next layer:
        for j in range(sizes[l+1]): # 70
            groupNum = int(j / nextGroupSize)  # assign this node to a group (indices starting at 0)
            startIndex = groupNum * groupSize
            #print('j = {}, group = {} -> {}:{}'.format(j, groupNum, startIndex, startIndex + groupSize))
            weightsMask[l][j][startIndex:(startIndex+groupSize)] = 1

        self.masks = weightsMask

        # zero out necessary weights:
        self.weights[self.special] = self.weights[self.special] * self.masks[self.special]


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
        #print("in mynetExperiment.py:updateMiniBatch")
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #print('nabla_b = ')
        #print(nabla_b)
        #print('nabla_w = ')
        #print(nabla_w)

        # average the delta_b, delta_w calculated for each training sample
        # and adjust the weights, biases using the learning rate
        for x, y in miniBatch:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb+db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw+dw for nw, dw in zip(nabla_w, delta_w)]

        # now apply mask to nabla_w (ensuring that zero'd out weights never change):
        nabla_w[self.special] = nabla_w[self.special] * self.masks[self.special]

        self.biases = [b-(rate/len(miniBatch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(rate/len(miniBatch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
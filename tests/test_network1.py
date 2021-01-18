"""
test_network1.py
~~~~~~~~~~~~~~~~

unit test mynet.py against network.py (the provided implementation)
"""

import pytest
import os
import sys
# enable imports from parent folder of this script:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import numpy as np
import random

import network
import mynet

from tests.conftest import get_dummy_training_data


def test_feed_forward():
    """
    test that feedforward() behaves the same across the two implemenations
    """
    for sizes in [ [4,2,3], [1,1], [10-x for x in range(10)], [(x+3)*((x%2)+3) for x in range(10)], [500, 3, 400, 1000] ]: 
        #sizes = [4,2,3]
        net = network.Network(sizes)
        # make a copy of this network using my class instead
        mine = mynet.Network(sizes)
        mine.weights, mine.biases, mine.sizes = copy.deepcopy(net.weights), copy.deepcopy(net.biases), copy.deepcopy(net.sizes)

        print("testing feed forward pass with network size: " + str(sizes))
        X, Y = get_dummy_training_data(sizes[0], sizes[-1], count=50)
        for x, y in zip(X, Y):
            zs, A = mine.feedforward(x)
            # my feedforward function additionally returns the Z value for each node
            assert(np.allclose(mine.sigmoid(zs[-1]), A[-1]))
            assert(np.allclose(mine.getOutput(x), A[-1]))
            assert(np.allclose(net.feedforward(x), mine.getOutput(x)))

def test_backprop():
    """
    test that backprop() behaves the same across the two implemenations
    """
    for sizes in [ [4,2,3], [1,1], [10-x for x in range(10)], [(x+3)*((x%2)+3) for x in range(10)], [100, 50, 150] ]: 
        net = network.Network(sizes)
        # make a copy of this network using my class instead
        mine = mynet.Network(sizes)
        mine.weights, mine.biases, mine.sizes = copy.deepcopy(net.weights), copy.deepcopy(net.biases), copy.deepcopy(net.sizes)

        print("testing backprop function with network size: " + str(sizes))
        X, Y = get_dummy_training_data(sizes[0], sizes[-1], count=25)
        for x, y in zip(X, Y):
            nabla_b, nabla_w = net.backprop(x, y)
            mine_b, mine_w = mine.backprop(x, y)
            # https://stackoverflow.com/a/30773738
            assert(all([np.allclose(nb, mb) for nb, mb in zip(nabla_b, mine_b)]))
            assert(all([np.allclose(nw, mw) for nw, mw in zip(nabla_w, mine_w)]))

def test_update_mini_batch():
    """
    test that update_mini_batch() behaves the same across the two implemenations
    """
    for sizes in [ [4,2,3], [1,1], [10-x for x in range(8)], [(x+3)*((x%2)+3) for x in range(7)], [100, 50, 20] ]: 
        net = network.Network(sizes)
        # make a copy of this network using my class instead
        mine = mynet.Network(sizes)
        mine.weights, mine.biases, mine.sizes = copy.deepcopy(net.weights), copy.deepcopy(net.biases), copy.deepcopy(net.sizes)

        print("testing update_mini_batch function with network size: " + str(sizes))
        X, Y = get_dummy_training_data(sizes[0], sizes[-1], count=25)
        batchSize = random.randint(1, 7)
        print("\tusing batchSize = {}".format(batchSize))

        while len(X) > 0:
            batch = [(x,y) for x,y in zip(X[:batchSize],Y[:batchSize])]
            X, Y = X[batchSize:], Y[batchSize:]

            iterations = 7 #random.randint(3, 10)
            eta = random.uniform(0.01, 0.25) # learning rate
            for i in range(iterations):
                net.update_mini_batch(batch, eta)
                mine.updateMiniBatch(batch, eta)
            # test weights and biases were updated to same values
            assert(all([np.allclose(b, mb) for b, mb in zip(net.weights, mine.weights)]))
            assert(all([np.allclose(w, mw) for w, mw in zip(net.biases, mine.biases)]))

        # also compare final outputs on some training data (after adjusting weights/biases)
        X, Y = get_dummy_training_data(sizes[0], sizes[-1], count=10)
        for x,y in zip(X, Y):
            assert(np.allclose(net.feedforward(x), mine.getOutput(x)))

def test_SGD():
    """
    test that SGD() behaves the same across the two implemenations
    """
    # TODO: finish this
    pass

"""
test_network1.py
~~~~~~~~~~~~~~~~

unit test mynet.py against network.py (the provided implementation)
"""

import pytest
from deepdiff import DeepDiff
import os, sys
# enable imports from parent folder of this script:
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
import copy
import numpy as np
import random
from timeit import default_timer as timer
from datetime import datetime

from mnist import mnist_loader
import network
import mynet

from tests.conftest import get_dummy_training_data


def test_feed_forward():
    """
    test that feedforward() behaves the same across the two implemenations
    """
    myTotalSec, refTotalSec = 0, 0
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

            start = timer()
            y_net = net.feedforward(x)
            refTotalSec += timer() - start

            start = timer()
            y_mine = mine.getOutput(x)
            myTotalSec += timer() - start

            assert(np.allclose(y_net, y_mine))
    print("reference: feedforward() cumulative time: {:.4f} sec".format(refTotalSec))
    print("mine:      feedforward() cumulative time: {:.4f} sec\n\n".format(myTotalSec))

def test_backprop():
    """
    test that backprop() behaves the same across the two implemenations
    """
    myTotalSec, refTotalSec = 0, 0
    for sizes in [ [4,2,3], [1,1], [10-x for x in range(10)], [(x+3)*((x%2)+3) for x in range(10)], [100, 50, 150] ]: 
        net = network.Network(sizes)
        # make a copy of this network using my class instead
        mine = mynet.Network(sizes)
        mine.weights, mine.biases, mine.sizes = copy.deepcopy(net.weights), copy.deepcopy(net.biases), copy.deepcopy(net.sizes)

        print("testing backprop function with network size: " + str(sizes))
        X, Y = get_dummy_training_data(sizes[0], sizes[-1], count=25)
        for x, y in zip(X, Y):
            start = timer()
            nabla_b, nabla_w = net.backprop(x, y)
            refTotalSec += timer() - start

            start = timer()
            mine_b, mine_w, _ = mine.backprop(x, y)
            myTotalSec += timer() - start
            # https://stackoverflow.com/a/30773738
            assert(all([np.allclose(nb, mb) for nb, mb in zip(nabla_b, mine_b)]))
            assert(all([np.allclose(nw, mw) for nw, mw in zip(nabla_w, mine_w)]))
    print("reference: backprop() cumulative time: {:.4f} sec".format(refTotalSec))
    print("mine:      backprop() cumulative time: {:.4f} sec\n\n".format(myTotalSec))

def test_update_mini_batch():
    """
    test that update_mini_batch() behaves the same across the two implemenations
    """
    myTotalSec, refTotalSec = 0, 0
    for sizes in [ [4,2,3], [1,1], [10-x for x in range(8)], [(x+3)*((x%2)+3) for x in range(7)], [100, 50, 20] ]: 
        net = network.Network(sizes)
        # make a copy of this network using my class instead
        mine = mynet.Network(sizes)
        mine.weights, mine.biases, mine.sizes = copy.deepcopy(net.weights), copy.deepcopy(net.biases), copy.deepcopy(net.sizes)

        X, Y = get_dummy_training_data(sizes[0], sizes[-1], count=25)
        batchSize = random.randint(1, 7)
        print("testing update_mini_batch function with network size: {}, batchSize: {}".format(sizes, batchSize))

        while len(X) > 0:
            batch = [(x,y) for x,y in zip(X[:batchSize],Y[:batchSize])]
            X, Y = X[batchSize:], Y[batchSize:]

            iterations = 7 #random.randint(3, 10)
            eta = random.uniform(0.01, 0.25) # learning rate
            for i in range(iterations):
                start = timer()
                net.update_mini_batch(batch, eta)
                refTotalSec += timer() - start

                start = timer()
                mine.updateMiniBatch(batch, eta)
                myTotalSec += timer() - start
            # test weights and biases were updated to same values
            assert(all([np.allclose(b, mb) for b, mb in zip(net.biases, mine.biases)]))
            assert(all([np.allclose(w, mw) for w, mw in zip(net.weights, mine.weights)]))

        # also compare final outputs on some training data (after adjusting weights/biases)
        X, Y = get_dummy_training_data(sizes[0], sizes[-1], count=10)
        for x,y in zip(X, Y):
            assert(np.allclose(net.feedforward(x), mine.getOutput(x)))
    print("reference: update_mini_batch() cumulative time: {:.4f} sec".format(refTotalSec))
    print("mine:      update_mini_batch() cumulative time: {:.4f} sec\n\n".format(myTotalSec))

def test_sigmoid_inverse():
    mine = mynet.Network([2, 2])
    for z in [0.38201, 5, 19.393939, -1.501]:
        res = mine.sigmoid(z)
        assert(np.isclose(z, mine.sigmoid_inverse(res)))

def test_SGD(tmp_path):
    """
    test that SGD() behaves the same across the two implemenations
    """
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)[:100]
    validation_data = list(validation_data)[:100]
    test_data = list(test_data)[:100]

    net = network.Network([784, 5, 10])
    mine = mynet.Network([784, 5, 10], backupDir=tmp_path)
    mine.weights, mine.biases, mine.sizes = copy.deepcopy(net.weights), copy.deepcopy(net.biases), copy.deepcopy(net.sizes)

    total_epochs, rate, mini_batch_size = 20, 3.0, 8
    myTotalSec, refTotalSec = 0, 0
    random.seed(0); np.random.seed # set random seed so behaviour is reproducible
    start = timer()
    net.SGD(training_data, epochs=total_epochs, mini_batch_size=mini_batch_size, eta=rate, test_data=test_data)
    refTotalSec += timer() - start

    random.seed(0); np.random.seed # ensure mini batches are shuffled the same as the last test
    start = timer()
    mine.SGD(training_data, epochs=total_epochs, mini_batch_size=mini_batch_size, rate=rate, test_data=test_data)
    myTotalSec += timer() - start
    print("reference: SGD took {:.4f} seconds".format(timer() - start))
    print("mine:      SGD took {:.4f} seconds\n\n".format(timer() - start))

    # test weights and biases were updated to same values
    #import pdb; pdb.set_trace()
    assert(all([np.allclose(b, mb) for b, mb in zip(net.biases, mine.biases)]))
    assert(all([np.allclose(w, mw) for w, mw in zip(net.weights, mine.weights)]))

def test_save_load(tmp_path):
    """
    verify saving/loading from file is functional.
    (tmp_path is a pytext fixture https://docs.pytest.org/en/6.2.x/tmpdir.html)
    """
    name = "unit_test--{}".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    net1 = mynet.Network([784, 5, 10], backupDir=tmp_path, name=name)
    print('tmp_path = ')
    print(tmp_path)

    net1.save('initial.pkl')
    saveDir = os.path.join(net1.backupDir, net1.name)
    assert(os.path.exists(os.path.join(saveDir, 'initial.pkl')))

    net2 = mynet.Network([1, 2], backupDir=tmp_path, name=name)
    net2.load('initial.pkl')

    assert(net1.backupDir == net2.backupDir)
    assert(net1.name == net2.name)

    d1 = net1.__dict__
    d2 = net2.__dict__
    # have to remove backupDir param for the diff to work as expected
    #   (TODO: understand this better)
    d1.pop('backupDir')
    d2.pop('backupDir')
    diff = DeepDiff(net1.__dict__, net2.__dict__)
    print('diff = ')
    print(diff)
    assert(bool(diff) == False)
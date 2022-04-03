"""
test_helpers.py
~~~~~~~~~~~~~~~~

unit test some helper methods in runGrammerTree.py
"""

import pytest
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

from runGrammarTree import joinNetworks, splitNetwork
from mnist import mnist_loader
import mynet as network


def test_joinNetworks():
  # TODO: split net0 into two, then test joining the two halves
  #   (the joined version should produce the same output for a given input)
  #net0 = network.Network([784, 70, 40, 15, 10])

  netA = network.Network([784, 70, 40])
  netB = network.Network([40, 15, 10])
  joined = joinNetworks([netA, netB])

  assert(joined.sizes == [784, 70, 40, 15, 10])
  for i in range(2):
    np.testing.assert_array_equal(joined.weights[i], netA.weights[i])
    np.testing.assert_array_equal(joined.biases[i], netA.biases[i])
  for i in range(2, 4):
    np.testing.assert_array_equal(joined.weights[i], netB.weights[i-2])
    np.testing.assert_array_equal(joined.biases[i], netB.biases[i-2])

  training_data = mnist_loader.load_data_wrapper()[0]
  x = training_data[0][0]
  joined.getOutput(x) # should not throw an error

  # newA should be equivalent to netA:
  newA = splitNetwork(joined, 2)[0]
  np.testing.assert_array_equal(netA.getOutput(x), newA.getOutput(x))


def test_splitNetwork():
  data =  mnist_loader.load_data_wrapper()[0]
  net0 = network.Network([784, 100, 70, 40, 15, 10])
  origOutput = net0.getOutput(data[0][0])

  net0A, net0B = splitNetwork(net0, 2)
  assert(net0A.sizes == [784, 100, 70])
  assert(net0B.sizes == [70, 40, 15, 10])

  newOutput = net0A.getOutput(data[0][0])
  newOutput = net0B.getOutput(newOutput)
  np.testing.assert_array_equal(origOutput, newOutput)

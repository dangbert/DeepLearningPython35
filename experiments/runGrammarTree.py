#!/usr/bin/env python3

import os, sys
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

import mynet as network
import numpy as np

import GrammarNet

from mnist import mnist_loader


def main():

  # TODO: try with a softmax on all hidden layers, and the output layer (so each layer sums to 1.0)
  #part1()
  part2()

def part2():
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  net0 = network.Network([784, 70, 40, 15, 10], name="net0", backupDir="backups/grammarTree")
  net1 = network.Network([784, 55, 40, 21, 10], name="net1", backupDir="backups/grammarTree")

  total_epochs = 5 # TODO for now
  rate, mini_batch_size = 3.0, 10

  net0.load('epoch5.pkl')
  net1.load('epoch5.pkl')

  # train initial network
  print("training net0 for {} epochs".format(total_epochs))
  net0.SGD(training_data, total_epochs, mini_batch_size, rate, test_data=test_data)
  print("\n------done training net0-----")

  print("training net1 for {} epochs".format(total_epochs))
  net1.SGD(training_data, total_epochs, mini_batch_size, rate, test_data=test_data)
  print("\n------done training net1-----")

  # convert to grammar nets
  print('converting net0 and net1 to grammar nets...')
  net0 = GrammarNet.Network([784, 70, 40, 15, 10], name="net0", backupDir="backups/grammarTree")
  net1 = GrammarNet.Network([784, 55, 40, 21, 10], name="net1", backupDir="backups/grammarTree")
  net0.load('epoch5.pkl')
  net1.load('epoch5.pkl')

  # subnet0_1 of net0 (starting at layer 2):
  #   (where this is now a network that takes input of size 40, and has output of size 10)
  #tmpNet = network.Network([40, 15, 10], name="tmpNet")
  #tmpNet.biases = net0.biases[3:]
  #tmpNet.weights = net0.weights[3:]


  # we add an extra layer from net1 for convenience for computing errors in the layer of size 40 with existing code...
  tmpNet = network.Network([55, 40, 15, 10], name="tmpNet")
  tmpNet.biases = net0.biases[1:] # so start at 1 instead of 2 (for that extra layer)
  tmpNet.weights = [net1.weights[1]] + net0.weights[2:] # borrow net1's weights for the first ("dummy") layer

  #training_dataTmp = [(x, net1.feedforward(x)[1][1]) for (x, _) in list(training_data)]
  #test_dataTmp =     [(x, net1.feedforward(x)[1][1]) for (x, _) in list(test_data)]

  print("special training of net1 ".format(total_epochs))
  # "activate" net1 as a proper GrammarNet, then continue training it
  net1.grammarLayer = 2
  net1.otherNets = [tmpNet]

  net1.SGD(training_data, total_epochs+25, mini_batch_size, rate, test_data=test_data)
  print("all done!")

  # now test to see if it will work (magically) in the opposite direction
  #   (feeding output of grammar layer in net0, into the final subnet of net1)
  #   (without any additional training)

  #net0 = GrammarNet.Network([784, 70, 40, 15, 10], name="net0", backupDir="backups/grammarTree")
  #net1 = GrammarNet.Network([784, 55, 40, 21, 10], name="net1", backupDir="backups/grammarTree")
  tmpNet2 = GrammarNet.Network([784, 70, 40, 21, 10], name="tmpNet2", backupDir="backups/grammarTree")
  tmpNet2.biases = net0.biases[:2] + net1.biases[2:]
  tmpNet2.weights = net0.weights[:2] + net1.weights[2:]

  # now evaluate this network on the test set (without training):

  # also try training net0 and net1 (as grammar trees)
  #   interleaving their epochs of training (and updating the respective tmp nets)
  #   to see if they can converge together on a working "grammar layer"



def part1():
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  net0 = network.Network([784, 100, 30, 10])
  total_epochs = 1 # TODO for now
  rate, mini_batch_size = 3.0, 10

  # train initial network
  print("training net0 for {} epochs".format(total_epochs))
  net0.SGD(training_data, total_epochs, mini_batch_size, rate, test_data=test_data)
  print("\n------done training net0-----")


  # now we can consider the layers in net0 to define a "Grammar" tree for this particular data...
  #  (with layers starting at index 0):
  #subnet0_0 = transfrom layer0 -> layer1
  subnet1_0 = network.Network([784, 100])
  #subnet1_0.biases = net0.biases[:1] # contain just first bias entry
  #subnet1_0.weights = net0.weights[:1]#

  #subnet0_1 = transfrom layer1 -> layer2
  subnet1_1 = network.Network([100, 30])

  #subnet0_2 = transfrom layer2 -> layer3
  subnet1_1 = network.Network([30, 10])

  # each of these subnets will be trained to emulate the transform (within net0) from layer l -> layer l+1
  #   net0 will be a teacher for these students

  # now create a dataset for the first subnet (using net0 as a teacher / source of ground truth)
  #   (we use the activations of layer 1 as the "label" / ground truth data to emulate later)

  # note feedforward() returns a tuple (zs, activations)
  training_data1_0 = [(x, net0.feedforward(x)[1][1]) for (x, _) in list(training_data)]
  test_data1_0 =     [(x, net0.feedforward(x)[1][1]) for (x, _) in list(test_data)]

  #(x, _) = training_data[0]
  #print("x.shape = {}".format(x.shape))
  #y = net0.feedforward(x)[1][1]

  print("\ncreated training/test datasets for subnet1_0")
  #import pdb; pdb.set_trace()
  #print("training data shape: ({}, {})".format(training_data1_0[0][0].shape, training_data1_0[0][1].shape))
  #print(training_data1_0[0])
  # now train subnet1_0 on this dataset from the "teacher"

  print("training subnet1_0 for {} epochs".format(total_epochs))
  subnet1_0.SGD(training_data1_0, total_epochs, mini_batch_size, rate, test_data=test_data1_0)
  # TODO: support test_data where a vector is provided as the ground truth rather than an index
  #subnet1_0.SGD(training_data1_0, total_epochs, mini_batch_size, rate)
  # NOTE: better to report the total cost on the test_data set than just the number of results that had the same highest activation index...
  print("-----\nall done!------")

  # next steps:
  # come up with way to swap out grammer layers or something
  #  to converge on a set of "standard" grammer layers for the networks.

  # perhaps create some networks with extra layers hidden between the grammer layers as part of this process...





  ####netLib = [network.Network([784, 100, 30, 10]) for _ in range(4)]

if __name__ == "__main__":
  main()
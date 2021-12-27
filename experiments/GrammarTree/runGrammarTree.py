#!/usr/bin/env python3

import os, sys
import matplotlib.pyplot as plt
import pickle
import copy

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))

import mynet as network
import numpy as np

import experiments.GrammarTree.GrammarNet as GrammarNet

from mnist import mnist_loader


def main():

  # TODO: try with a softmax on all hidden layers, and the output layer (so each layer sums to 1.0)
  #part1()
  part2()

BACKUP_DIR = "backups/grammarTree"
STATS_PATH = "{}/stats".format(BACKUP_DIR)

def part2():
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  net0 = network.Network([784, 70, 40, 15, 10], name="net0", backupDir=BACKUP_DIR)
  net1 = network.Network([784, 55, 40, 21, 10], name="net1", backupDir=BACKUP_DIR)
  GL = 2 # grammar layer index

  total_epochs = 5
  rate, mini_batch_size = 3.0, 10

  """
  # train initial networks
  print("training net0 for {} epochs".format(total_epochs))
  net0.SGD(training_data, total_epochs, mini_batch_size, rate, test_data=test_data)
  print("\n------done training net0-----")
  print("training net1 for {} epochs".format(total_epochs))
  net1.SGD(training_data, total_epochs, mini_batch_size, rate, test_data=test_data)
  print("\n------done training net1-----")
  """

  # convert to grammar nets
  print('creating mainPool of grammer nets')
  mainPool = [
    GrammarNet.Network([784, 70, 40, 15, 10], name="net0", backupDir=BACKUP_DIR),
    GrammarNet.Network([784, 55, 40, 21, 10], name="net1", backupDir=BACKUP_DIR),
    GrammarNet.Network([784, 100, 50, 40, 21, 10], name="net2", backupDir=BACKUP_DIR),
    GrammarNet.Network([784, 55, 40, 30, 15, 10], name="net3", backupDir=BACKUP_DIR),
  ]

  #net0.load('epoch5.pkl')
  #net1.load('epoch5.pkl')
  #net0.load('latest.pkl')
  #net1.load('latest.pkl')
  #net0.backupDir, net1.backupDir = BACKUP_DIR, BACKUP_DIR # in case it's different in the pkl file

  #net1.grammarLayer, net0.grammerLayer = GL, GL
  #net1.otherNets = GL, [tmp0]
  #net0.grammarLayer, net0.otherNets = GL, [tmp1]

  stats = { 'epochs': [], 'pool vote': [] }

  for n in mainPool:
    stats[n.name] = []

  if os.path.exists(STATS_PATH + '.pkl'):
    with open(STATS_PATH + '.pkl', 'rb') as f:
      stats = pickle.load(f)
      print('loaded existing stats from file')

  # "activate" net1 as a proper GrammarNet, then continue training it
  for n in mainPool:
    n.grammarLayer = n.sizes.index(40) # find by grammar layer size

  curEpoch, total_epochs = mainPool[0].epoch, 1000
  updateStats(mainPool, stats, test_data)
  plotStats(stats, xlabel="Epoch", ylabel="Accuracy (test set) %", title="Experiment2c")
  print("grammar training of mainPool ({} total epochs)".format(total_epochs))
  for e in range(curEpoch+1, total_epochs+1, 3):
    # TODO: show total of (modified) cost function during training
    #   (C = C_net0 + C_tmp1)

    for i in range(len(mainPool)):
      mainPool[i].otherNets = [splitNetwork(mainPool[k], mainPool[k].grammarLayer)[1] for k in range(len(mainPool)) if k != i]
      mainPool[i].SGD(training_data, e, mini_batch_size, rate, test_data=test_data)

    updateStats(mainPool, stats, test_data)
    plotStats(stats, xlabel="Epoch", ylabel="Accuracy (test set) %", title="Experiment2c")
  print("all done!")


  # TODO: next steps:
  # [ ] optimize interleaved training, when computing additional blame we my as well as update the weights in the tmp network...
  # [X] generalize code for training a pool of 3+ networks with a shared grammar layer...
  # [X] test having subnetworks generated from the pool and voting on the final outpout

def joinNetworks(nets):
  """
  Combines a provided list of networks.
  The size of the output layer of a given network, must equal the size of the input layer of the next network in nets.

  Args:
    nets (array of Network objects): list of networks to be combined into one
  """
  final = copy.deepcopy(nets[0])
  for n in nets[1:]:
    final.sizes = final.sizes[:-1] + n.sizes
    final.weights += copy.deepcopy(n.weights)
    final.biases += copy.deepcopy(n.biases)
  return final

def splitNetwork(net, splitLayer, name=None):
  """
  return two networks, formed by splitting the provided network into two (at the given layer).
    the given layer will be both the output layer of the first network returned, and the input layer of the second.

  Args:
    net (:obj Network): Network to split into two
    splitLayer (int): index of layer in net to split
  """
  name = 'tmp-' + net.name if name == None else name
  netB = network.Network(net.sizes[splitLayer:], name=name)
  netB.biases = copy.deepcopy(net.biases[splitLayer:])
  netB.weights = copy.deepcopy(net.weights[splitLayer:])

  netA = network.Network(net.sizes[:(splitLayer+1)], name=name)
  netA.biases = copy.deepcopy(net.biases[:splitLayer])
  netA.weights = copy.deepcopy(net.weights[:splitLayer])
  return netA, netB

def evaluatePool(nets, test_data, median=False):
  """
  evaluate a pool of networks by having them vote on the final output of the test_data.
  returns the percent of the test_data for which the pool predicted the correct output.
  """

  # TOOD: experiment with combinig inferences as medians vs averages...
  outputs = [[net.getOutput(x) for net in nets] for x, _ in test_data]
  inferences = []
  for data in outputs:
    if median:
      inferences.append(np.median(data, axis=0))
    else:
      # average all np arrays in data, and store the result array
      inferences.append(np.average(data, axis=0))

  # based on mynet.py:test()
  ys = [y for _, y in test_data]
  test_results = [(np.argmax(x), y)
                  for (x, y) in zip(inferences, ys)]
  correct = sum(int(x == y) for (x, y) in test_results)
  return 100 * correct / len(test_data)

def updateStats(mainPool, stats, test_data):
  stats['epochs'].append(mainPool[0].epoch)
  for n in mainPool:
    stats[n.name].append(100 * n.test(test_data) / len(test_data))

  # create a larger pool of networks from all combinations of mainPool (splitting each at its grammar layer)
  half1 = []
  half2 = []
  for i in range(len(mainPool)):
    h1, h2 = splitNetwork(mainPool[i], mainPool[i].grammarLayer)
    half1.append(h1)
    half2.append(h2)

  votePool = []
  for i in range(len(half1)):
    for j in range(len(half2)):
      votePool.append(joinNetworks([half1[i], half2[j]]))

  for i in range(len(votePool)):
    n = votePool[i]
    print("votePool[{}] performance: {:.2f}%".format(i, 100 * n.test(test_data) / len(test_data)))
  res = evaluatePool(votePool, test_data)
  print("combined pool performance:   {:.2f}%".format(res))
  stats['pool vote'].append(res)

def evaluate(n0, tmp, test_data, gl):
  """
  feed outputs of grammar layer gl in net n0 as input into network tmp and evaluate its performance on test_data
  Args:
    gl (int): index of Grammary layer in net n0
  """
  print('\nfeeding grammer layer from "{}" as inputs into "{}"...'.format(n0.name, tmp.name))
  print('respective net sizes are: {}, and  {}'.format(n0.sizes, tmp.sizes))
  xs = [n0.feedforward(x)[1][gl] for x, _ in test_data]
  ys = [y for x, y in test_data]
  data = [(x, y) for (x, y) in zip(xs, ys)]
  #import pdb; pdb.set_trace()
  correct = tmp.test(data)
  print('correct = {} / {} = {:.2f}%\n'.format(correct, len(data), 100 * correct / len(data)))
  return 100 * correct / len(data)

def plotStats(stats, show=False, statsPath=STATS_PATH, xlabel="", ylabel="", title=""):
  """
  plot stats of the format:
    { 'epochs': [], 'net0': [], 'tmp1': [], 'tmp0': [] }

  TODO: create a class for tracking, storing generalized training stats...
  """
  plt.clf()
  for key in stats.keys():
    if key == 'epochs':
      continue
    plt.plot(stats['epochs'], stats[key], label=key)

  plt.legend()
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  if show:
    plt.show()
  plt.savefig("{}.png".format(statsPath), dpi=400)
  with open("{}.pkl".format(statsPath), 'wb') as f:
    pickle.dump(stats, f, 2)


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
#!/usr/bin/env python3

import os, sys
import matplotlib.pyplot as plt
import pickle

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

  def createTmpNet(n0, gL=2, name=None):
    """
    Create a new network defined by the subnet starting at the grammar layer in net n0, and continuing to the end.

    Args:
      gl (int): index of layer in net n0 to act as the "grammarLayer"
      name (str): name for the new network
    """
    tmp = network.Network(n0.sizes[gL:], name=name)
    tmp.biases = n0.biases[gL:]
    tmp.weights = n0.weights[gL:]
    return tmp

  # convert to grammar nets
  print('converting net0 and net1 to grammar nets...')
  net0 = GrammarNet.Network([784, 70, 40, 15, 10], name="net0", backupDir=BACKUP_DIR)
  net1 = GrammarNet.Network([784, 55, 40, 21, 10], name="net1", backupDir=BACKUP_DIR)
  net0.load('epoch5.pkl')
  net1.load('epoch5.pkl')
  net0.name, net1.name = 'gnet0', 'gnet1'
  net0.load('latest.pkl')

  # if we feed the activations from the grammary layer in net0 into the subnet tmp
  #   we should get the same output from both networks
  x, _ = test_data[0]
  tmp0 = createTmpNet(net0, name='tmp0')
  _, actsOrig = net0.feedforward(x)
  assert(np.allclose(actsOrig[-1], tmp0.feedforward(actsOrig[2])[1][-1]))
  # test again using a different network
  tmp1 = createTmpNet(net1, name='tmp1')
  _, actsOrig = net1.feedforward(x)
  assert(np.allclose(actsOrig[-1], tmp1.feedforward(actsOrig[2])[1][-1]))


  #net0.load('backups/grammarTree/net0_experiment0/epoch95.pkl')
  def evaluate(n0, tmp, test_data):
    print('\nfeeding grammer layer from "{}" as inputs into "{}"...'.format(n0.name, tmp.name))
    print('respective net sizes are: {}, and  {}'.format(n0.sizes, tmp.sizes))
    xs = [n0.feedforward(x)[1][2] for x, _ in test_data]
    ys = [y for x, y in test_data]
    data = [(x, y) for (x, y) in zip(xs, ys)]
    #import pdb; pdb.set_trace()
    correct = tmp.test(data)
    print('correct = {} / {} = {:.2f}%\n'.format(correct, len(data), 100 * correct / len(data)))
    return 100 * correct / len(data)

  def evaluate2(n0, n1, test_data, gl=2):
    """
    also evaluate going the opposite direction
    feeding output of grammer layer in net n1 into the final subnetwork of n0
    """
    tmp0 = createTmpNet(n0, name="tmp-{}".format(n0.name))
    return evaluate(n1, tmp0, test_data)



  print("grammar training of net0".format(total_epochs))
  curEpoch, total_epochs = net0.epoch, 3000
  # "activate" net1 as a proper GrammarNet, then continue training it
  net1.grammarLayer = 2
  net0.grammarLayer = 2
  net0.otherNets = [tmp1]
  net1.otherNets = [tmp0]

  stats = {
    'epochs': [],
    'net0': [],
    'tmp1': [],
    'tmp0': [],
  }
  with open(STATS_PATH + '.pkl', 'rb') as f:
    stats = pickle.load(f)


  plotStats(stats, xlabel="Epoch", ylabel="Accuracy (test set) %", title="Experiment2")
  for i in range(curEpoch+1, total_epochs+1, 5):
    #print("\ntraining both nets on epoch {}".format(i))
    # TODO: show total of (modified) cost function during training
    #   (C = C_net0 + C_tmp)
    net0.SGD(training_data, i, mini_batch_size, rate, test_data=test_data)

    stats['epochs'].append(net0.epoch)
    stats['net0'].append(100 * net0.test(test_data) / len(test_data))
    stats['tmp1'].append(evaluate(net0, tmp1, test_data))
    stats['tmp0'].append(evaluate2(net0, net1, test_data, gl=2))
    plotStats(stats, xlabel="Epoch", ylabel="Accuracy (test set) %", title="Experiment2")

    #net1.otherNets = [createTmpNet(net0)]
    #net1.SGD(training_data, i, mini_batch_size, rate, test_data=test_data)

  # after training just net0, test to see if it will work (magically) in the opposite direction:
  #   (feeding output of grammar layer in net0, into the final subnet of net1):


  print("all done!")

  # TODO: next steps:
  # [ ] try training net0 and net1 (as grammar trees) interleaving their epochs of training
  #     (and updating the respective tmp nets)
  #     to see if they can converge together on a working "grammar layer"
  # [ ] optimize interleaved training, when computing additional blame we my as well as update the weights in the tmp network...
  # [ ] generalize code for training a pool of 3+ networks with a shared grammar layer...
  # [ ] test having subnetworks generated from the pool and voting on the final outpout


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
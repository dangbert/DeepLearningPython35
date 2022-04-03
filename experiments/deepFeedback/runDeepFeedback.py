#!/usr/bin/env python3

import os, sys
import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))

import mynet as network
from mnist import mnist_loader
import experiments.deepFeedback.DeepFeedNet as DNetwork
from experiments.GrammarTree.runGrammarTree import plotStats


BACKUP_DIR = "backups/deepFeedback"
STATS_PATH = "{}/stats".format(BACKUP_DIR)

def main():

  # train networks
  #part1()
  # plot results (using backups from disk)
  stats = getStats(200)
  import pdb; pdb.set_trace()
  statsPath = os.path.join(BASE_DIR, "archive/experiment1/stats")
  plotStats(stats, show=True, statsPath=statsPath, xlabel="epochs", ylabel="% correct (test set)", title="experiment 1, deep net vs control")


def part1():
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

  # dimensions of feedback layer...
  FEEDBACK_DIM = 15

  # "normal network" to compare experimental networks' performance to
  sizes = [784, 70, 40, FEEDBACK_DIM, 10]
  netControl = network.Network(sizes, name="netControl", backupDir=BACKUP_DIR)

  # experimental network
  #sizes = netControl.sizes
  #sizes[0] != FEEDBACK_DIM
  net0 = DNetwork.Network(sizes, name="net0", backupDir=BACKUP_DIR)

  evaluateNet(netControl)
  evaluateNet(net0)

  # now train both networks:
  # TODO: interleave training so it's more "parallel" (see runGramarTree.py)
  total_epochs = 200
  rate, mini_batch_size = 3.0, 10
  net0.SGD(training_data, total_epochs, mini_batch_size, rate, test_data=test_data)
  netControl.SGD(training_data, total_epochs, mini_batch_size, rate, test_data=test_data)

  print("\nDone training networks! evaluating...")
  evaluateNet(netControl)
  evaluateNet(net0)

# TODO: extract this method into a common location for shared usage?
def getStats(totalEpochs):
  """load backups of networks, and return stats about their performance"""

  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  netControl = network.Network([1,2,3], name="netControl", backupDir=BACKUP_DIR)
  net0 = DNetwork.Network([1,2,3], name="net0", backupDir=BACKUP_DIR)

  stats = { "epochs": [], "net0": [], "netControl": [] }
  for e in range(0, totalEpochs+1, 25):
    fname = "epoch{}.pkl".format(str(e).rjust(4, '0')) # 0 pad epoch
    if e == 0:
      fname = "initial.pkl"
    print(fname)

    stats["epochs"].append(e)

    net0.load(fname)
    stats["net0"].append(evaluateNet(net0, print=False))

    netControl.load(fname)
    stats["netControl"].append(evaluateNet(netControl, print=False))
  return stats


def evaluateNet(net, print=True):
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  #for (x,y) in training_data

  test_data = list(test_data)

  correctCount = net.test(test_data)

  if print:
    print(f"net {net.name} :\t{correctCount}/{len(test_data)} = {(correctCount/len(test_data)):.3f} (test set performance)")
  #import pdb; pdb.set_trace()
  return correctCount / len(test_data)

if __name__ == "__main__":
  main()

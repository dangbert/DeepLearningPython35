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


#BACKUP_DIR = "backups/deepFeedback"
BACKUP_DIR = "backups/deepFeedback_part1b"
#STATS_PATH = "{}/stats".format(BACKUP_DIR)

STATS_PATH = os.path.join(BASE_DIR, "archive/experiment1b/stats")

def main():

  # train networks
  #part1b(total_epochs=40)
  #part1b(total_epochs=1330, resumeFromEpoch=1327)

  # plot results (using backups from disk)
  stats, net0 = loadPrevRun(STATS_PATH, "epoch1327.pkl")
  plotPart1b(stats, STATS_PATH, net0, totalIter=40)


def loadPrevRun(statsPath, pklName, netName="net0"):
  net = DNetwork.Network([1,2,3], name=netName, backupDir=BACKUP_DIR)
  net.load(pklName)

  stats = { "epochs": [], netName: [] }
  if os.path.exists(STATS_PATH + '.pkl'):
    with open(STATS_PATH + '.pkl', 'rb') as f:
      stats = pickle.load(f)
      print('loaded existing stats from file')
  return stats, net
  
def part1b(total_epochs, resumeFromEpoch=None):
  """
  Same as part 1, except now seeding first iteration with gaussian noise, and using 4 iterations for training.
  Note: number of iterations to do when training is determined by the default value of totalIter in DeepFeedNet:feedforward()
  """
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

  # dimensions of feedback layer...
  FEEDBACK_DIM = 15
  EVAL_ITERS = 4 # number of iterations to use at test/evaluation stage

  # "normal network" to compare experimental networks' performance to
  sizes = [784, 70, 40, FEEDBACK_DIM, 10]
  #netControl = network.Network(sizes, name="netControl", backupDir=BACKUP_DIR)

  # experimental network
  #sizes = netControl.sizes
  #sizes[0] != FEEDBACK_DIM
  net0 = DNetwork.Network(sizes, name="net0", backupDir=BACKUP_DIR)

  #evaluateNet(netControl, print=True)

  # now train networks:
  stats = { "epochs": [], "net0": [] }
  rate, mini_batch_size = 3.0, 10
  curEpoch = 0

  # load prev data from disk if applicable:
  if resumeFromEpoch is not None:
    curEpoch = resumeFromEpoch
    pklName = f"epoch{zeroPad(resumeFromEpoch, 4)}.pkl"
    print(f"resuming from: {pklName}")

    stats, net0 = loadPrevRun(STATS_PATH + '.pkl', pklName)

  evaluateNet(net0, verbose=True, totalIter=EVAL_ITERS)

  for e in range(curEpoch+1, total_epochs+1, 4):
    net0.SGD(training_data, e, mini_batch_size, rate, test_data=test_data)
    #netControl.SGD(training_data, e, mini_batch_size, rate, test_data=test_data)
    stats["epochs"].append(e)
    stats[net0.name].append(evaluateNet(net0, totalIter=EVAL_ITERS))

    pklName = f"epoch{zeroPad(e, 4)}.pkl"
    #plotPart1b(stats, STATS_PATH, pklName=pklName) # if we want to test over varying total iterations as well
    plotPart1b(stats, STATS_PATH)

  print("\nDone training networks! evaluating...")
  #evaluateNet(netControl, verbose=True)
  evaluateNet(net0, verbose=True, totalIter=EVAL_ITERS)

  # plot/save stats
  #statsPath = os.path.join(BASE_DIR, "archive/experiment1b/stats")
  #pklName = f"epoch{zeroPad(e, 4)}.pkl"
  plotPart1b(stats, STATS_PATH, net0)

  #plotStats(stats, show=True, statsPath=statsPath + "_iterations", xkey="iterations", xlabel="total iterations (per input)", ylabel="% correct (test set)", title="experiment 1, net0 (epoch 200) performance over varying (total) iterations")



def plotPart1b(stats, statsPath, net=None, totalIter=15):
  """
  plot results from part1b()
  params:
    stats: stats from experiment
    pklName: e.g. "epoch0200" (if not set, we won't test effects of varying total iterations during inference)
  """

  # plot stats from training:
  plotStats(stats, show=False, statsPath=statsPath, xlabel="epochs", ylabel="% correct (test set)", title="experiment 1b, deep feedback net")

  if net is None:
    return

  # see effect of different totalIter values when testing
  stats = {
    "iterations": [],
    net.name: [],
  }
  for total in range(1, totalIter):
    stats["iterations"].append(total)
    stats[net.name].append(evaluateNet(net, totalIter=total))
    plotStats(stats, show=False, statsPath=statsPath + "_iterations", xkey="iterations", xlabel="total iterations (per input)", ylabel="% correct (test set)", title=f"experiment 1b, {net.name} (epoch {net.epoch}) performance over varying iterations")

  #plotStats(stats, show=False, statsPath=statsPath + "_iterations", xkey="iterations", xlabel="total iterations (per input)", ylabel="% correct (test set)", title=f"experiment 1b, {net.name} (epoch {net.epoch}) performance over varying iterations")
  #return

  # if we didn't save stats while training, we can recover some of them by reading/testing backup files:
  #stats = getStats(200)
  #plotStats(stats, show=True, statsPath=statsPath, xlabel="epochs", ylabel="% correct (test set)", title="experiment 1, deep net vs control")



# TODO: extract this method into a common location for shared usage?
#  (or at least make some helpers to return the available data for a given network's backup dir...)
def getStats(totalEpochs):
  """
  load backups of networks, and return stats about their performance
  (used if you didn't track/save stats while training the network).
  """

  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  netControl = network.Network([1,2,3], name="netControl", backupDir=BACKUP_DIR)
  net0 = DNetwork.Network([1,2,3], name="net0", backupDir=BACKUP_DIR)

  stats = { "epochs": [], "net0": [], "netControl": [] }
  for e in range(0, totalEpochs+1, 25):
    fname = "epoch{}.pkl".format(zeroPad(e, 4)) # 0 pad epoch
    if e == 0:
      fname = "initial.pkl"
    print(fname)

    stats["epochs"].append(e)

    net0.load(fname)
    stats["net0"].append(evaluateNet(net0, print=False))

    netControl.load(fname)
    stats["netControl"].append(evaluateNet(netControl, print=False))
  return stats

# TODO: move to general helper lib somewhere
def zeroPad(num, width=4):
  """
  returns provided number (int) as a zero padded string of desired width
  """
  return str(num).rjust(width, '0')



def evaluateNet(net, verbose=False, totalIter=None):
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  #for (x,y) in training_data

  test_data = list(test_data)

  if totalIter is None:
    correctCount = net.test(test_data)
  else:
    correctCount = net.test(test_data, totalIter=totalIter)

  if verbose:
    print(f"net {net.name} :\t{correctCount}/{len(test_data)} = {(correctCount/len(test_data)):.3f} (test set performance)")
  #import pdb; pdb.set_trace()
  return correctCount / len(test_data)

if __name__ == "__main__":
  main()

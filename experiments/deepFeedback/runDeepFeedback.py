#!/usr/bin/env python3

import os, sys
import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))

import mynet as network
from mnist import mnist_loader
import experiments.deepFeedback.DeepFeedNet as DNetwork


BACKUP_DIR = "backups/deepFeedback"
STATS_PATH = "{}/stats".format(BACKUP_DIR)

def main():

  part1()


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

  evaluateExNet(netControl)
  evaluateExNet(net0)



def evaluateExNet(net):
  """evaluate an experimental (deep feeback) net"""
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  #for (x,y) in training_data

  test_data = list(test_data)

  correctCount = net.test(test_data)

  print(f"net {net.name} :\t{correctCount}/{len(test_data)} = {(correctCount/len(test_data)):.3f} (test set performance)")
  #import pdb; pdb.set_trace()

if __name__ == "__main__":
  main()

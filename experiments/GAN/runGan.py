#!/usr/bin/env python3
import os, sys, argparse
import matplotlib.pyplot as plt
import pickle
import copy

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))

import mynet as network
import numpy as np

#import experiments.GrammarTree.GrammarNet as GrammarNet
from mnist import mnist_loader

BACKUP_DIR = "backups/GAN"
STATS_PATH = "{}/stats".format(BACKUP_DIR)

def main():
    parser = argparse.ArgumentParser(description='Builds configures and starts personalpedia locally')
    parser.add_argument('-s', '--start-epoch', required=False, type=int, default=-1, help='epoch to start at (default 1)')
    parser.add_argument('-e', '--end-epoch', required=False, type=int, default=15, help='epoch to end at (default 15)')
    args = parser.parse_args()


    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    rate, mini_batch_size = 3.0, 10

    # load/train classifier network
    net0 = network.Network([784, 70, 40, 15, 10], name="classifer", backupDir=BACKUP_DIR)
    if args.start_epoch != -1:
        net0.load(f"epoch{zeroPad(args.start_epoch)}.pkl")
    if net0.epoch < args.end_epoch:
      net0.SGD(training_data, args.end_epoch, mini_batch_size, rate, test_data=test_data)

    print(f"classifer ready (at epoch {net0.epoch})")

    #target = np.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0])
    target = np.zeros((10, 1))
    target[2] = 1

    input = generateData(net0, target)
    mnist_loader.vectorToImage(input).save("out.png")
    # TODO: report confidence of network on image's classification...
    print("done! wrote out.png")



def generateData(net: network.Network, desiredOutput: np.ndarray):
    """Given a trained network, generate an input that produces the desiredOutput.
    (like an adversarial attack).
    """
    # initial (random) input vector
    input = np.random.randn(net.sizes[0], 1)

    iter, maxIter = 0, 100
    rate = 3
    errors = []
    # TODO: train until error converges
    print("generating synthetic input...")
    while True:
        input[input<0.0] = 0.0
        input[input>1.0] = 1.0
        if iter >= maxIter:
            break

        (_, _, nabla_a0) = net.backprop(input, desiredOutput)
        error = net.testQuadraticCost([(input, desiredOutput)])
        errors.append(error)

        # update input using backprop
        input = input - rate * nabla_a0
        iter += 1
    return input

def zeroPad(num, digits=4):
    return str(num).rjust(digits, '0') # 0 pad number

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

# ----------------------
# - read the input data:
from mnist import mnist_loader
import os
import glob
import numpy as np
import shutil

# ---------------------
# test network.py
#import network
import mynet as network
#import costPrediction as network

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    train()
    visualizeResults(test_data)
    #test()

# visualize results by outputing them as images to a file
# dataset = list of (x,y) pairs
# fname = path to pkl file
# saveSucccesses (if we should save correct predictions to images as well)
def visualizeResults(dataSet, fname="backups/latest.pkl", saveSuccesses=False):
    net = network.Network([784, 30, 10])
    net.load(fname)

    print("testing provided dataset on network " + fname)
    index = 0
    correct = 0

    if os.path.exists("results/"):
        shutil.rmtree("results/")

    for (x, y) in dataSet:
        result = np.argmax(net.getOutput(x))
        status = "passed" if result==y else "failed"
        path = "results/" + status + "/" + str(result)
        if result != y or saveSuccesses == True:
            if not os.path.exists(path):
                os.makedirs(path)
            mnist_loader.vectorToImage(x, path + "/" + str(index) + "--correct-" + str(y) + ".png")
        index += 1
        correct += 1 if result==y else 0
    print("result: " + str(correct) + "/" + str(len(dataSet)))
    print("       (" + str(100.0 * correct / len(dataSet)) + "%)")

def train():
    # load dataset
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    #print("Training data:\t {} items".format(len(training_data)))
    #print("Validation data: {} items".format(len(validation_data)))
    #print("Test data:\t {} items".format(len(test_data)))

    start_epoch, total_epochs = 1, 15
    #start_epoch, total_epochs = 15, 150
    rate, mini_batch_size = 3.0, 10

    # train new network
    #net = network.Network([784, 30, 20, 10])   # about 95% accuracy (after 15 epoch)
    #net = network.Network([784, 100, 30, 10])  # about 96% accuracy (after 40 epoch)
    net = network.Network([784, 30, 10])        # about 95% accuracy (after 10 epochs)
    if start_epoch != 1:
        net.load("backups/epoch" + str(start_epoch) + ".pkl")

    # start training
    net.SGD(training_data, total_epochs, mini_batch_size, rate, test_data=test_data, start_epoch=start_epoch)

def test():
    # load dataset
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    test_data = list(test_data)
    training_data = list(training_data)

    # load old network from file and test
    print("testing on old networks")

    search_dir="backups"
    files = list(filter(os.path.isfile, glob.glob(search_dir + "/*.pkl")))
    files.sort(key=lambda x: os.path.getmtime(x))

    for fname in files:
        net = network.Network([784, 30, 10])
        net.load(fname)
        n_test = len(test_data)
        n_correct = net.test(test_data)
        print("{}\t: {:.2f}% -- {} / {}".format(fname, (n_correct/n_test)*100, n_correct, n_test))
    
if __name__ == "__main__":
    main()

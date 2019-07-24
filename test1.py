#!/usr/bin/env python3

# ----------------------
# - read the input data:
import mnist_loader
import os
import glob
import numpy as np

# ---------------------
# test network.py
#import network
import mynet as network

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #visualizeResults(test_data)

    train()
    #test()

# visualize results by outputing them to a file
# dataset = list of (x,y) pairs
# fname = path to pkl file
# saveSucccesses (if we should save correct predictions to images as well)
def visualizeResults(dataSet, fname="backups/latest.pkl", saveSuccesses=False):
    net = network.Network([784, 30, 10])
    net.load(fname)

    print("testing provided dataset on network " + fname)
    index = 0
    correct = 0
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

    start_epoch = 1
    total_epochs = 1000
    rate = 3.0
    mini_batch_size = 10

    # train new network
    net = network.Network([784, 30, 10])
    #net = network.Network([784, 5, 10])
    if start_epoch != 1:
        net.load("backups/epoch" + str(start_epoch) + ".pkl")

    # start training
    net.SGD(training_data, total_epochs, mini_batch_size, rate, test_data=test_data, start_epoch=start_epoch)

def test():
    # load dataset
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)

    # load old network from file and test
    print("testing on old networks")

    search_dir="backups"
    files = list(filter(os.path.isfile, glob.glob(search_dir + "/*.pkl")))
    files.sort(key=lambda x: os.path.getmtime(x))

    for fname in files:
        net = network.Network([784, 30, 10])
        net.load(fname)
        res = net.evaluate(list(test_data))
        print("{}\t-> {} / {}".format(fname, res, len(list(test_data))))
    
if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# ----------------------
# - read the input data:
import mnist_loader
import os
import glob

# ---------------------
# test network.py
import network

def main():

    train()
    #test()


def train():
    # load dataset
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)

    print("Training data: {}".format(len(training_data)))
    print("Validation data: {}".format(len(validation_data)))
    print("Test data: {}".format(len(test_data)))

    start_epoch = 1
    total_epochs = 5000

    # train new network
    net = network.Network([784, 30, 10])
    if start_epoch != 1:
        net.load("backups/epoch" + start_epoch + ".pkl")

    # start training
    net.SGD(training_data, total_epochs, 10, 3.0, test_data=test_data, start_epoch=start_epoch)


def test():
    # load dataset
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)

    # load old network from file and test
    print("testing on old networks")
    # TODO: test that save() load() actually works...

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

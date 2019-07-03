#!/usr/bin/env python3

# ----------------------
# - read the input data:
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# ---------------------
# test network.py
import network


# load old network from file and test
print("testing on old networks")
# TODO: test that save() load() actually works...
for i in range(0, 25, 5):
    fname = "backups/epoch" + str(i) + ".pkl"
    net = network.Network([784, 30, 10])
    net.load(fname)
    res = net.evaluate(list(test_data))
    print("{}\t-> {} / {}".format(fname, res, len(list(test_data))))


exit(0)


# train new network
# TODO: why does it appear to do so well immediately?
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
print("everything done")


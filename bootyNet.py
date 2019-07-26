#!/usr/bin/env python3
# bootyNet = "bootstrapping netwok"

# ----------------------
# - read the input data:
import mnist_loader
import os
import glob
import numpy as np

# ---------------------
# test network.py
#import network
#import mynet as network
import trueOrigNetwork as network
# for stealing its save and load functions
import mynet as helper

# TODO: probably will want to be able to store a universe object
# in some clean way (that can be recursively bootstrapped)
# you know the final universe has truly learned an idea if all the images
#  it generates look real images of the concept (e.g. "cat") to you.
def main():

    # keep in mind that when you do run out of resources, ...? I forget.
    # I was gonna save to keep training/optimize the new one
    # idk maybe do something smart with the set of all universes that existed
    #  (the best performing one taken from each's lifetime)
    #   oh and perhaps whenever a big-bang happens, we can dilute the old
    #   groundtruth dataset a bit using the new synthesized data.
    #   e.g. give a certain weight to the older data, and another weight to the new (syntesized) "ground-truth-ish" test set
    #
    # also instead of just adding a new layer each time, we could also/instead make the input images a bit larger
    # over time creating exponentially bigger photos in the datasets
    # (this would help us visualize a high-quality projection of what it
    # thinks a certain concept looks like.
    gooo() # :)

# oh yes, buckle up please
# prev = last known universe
# entropyYears = how many universes have existed
def gooo(prev=None, entropyYears=0):
    orig_training_data, orig_validation_data, orig_test_data = mnist_loader.load_data_wrapper()
    print("going")

    if prev == None:
        # this is the first universe to be born :)
        test_data = orig_test_data
        cur = network.Network([784, 30, 10]) # the universe as we know it (oh
                                             # also we have a huge pile of traininging-data "dust"

    # TODO: may not be sure how to get one network to make a guess at what a "cat" looks like...
    #       (reread my notes, or learn about GANs, or learn how to
    #       tweak an image of static more towards a "cat" as measured by cat detector network.
    # pool of unit-level networks allowed in our universe
    # TODO: something with an object for holding a universe of recursive members
    # (e.g. a universe of other comonent-universes that are smaller)
    pool = []
    pool.append(cur)

    history = {} # map epoch (int) to its performace percentage there on the *test_data*
    maxEpochs = 5            # epoch step size before re-evaluating if it has converged
    miniBatchSize = 10       # hardcoded, low number for now
    threshold = 0.005
    while getConvergenceRate(history) > threshold:
        if miniBatchSize < 100:              # hardcoded limit on bathsize
            miniBatchSize *= 1.1             # slowly increase batch size over time

        # pull the ignition
        cur.SGD(orig_training_data, maxEpochs, miniBatchSize, rate, test_data=test_data)
        score, test_results = cur.evaluate(test_data)
        # cur.epochs is total cumulative number of epochs its been subjected to
        history[cur.epochs] = score
        print("score: " + score + "\tnet.epochs = " + str(net.epochs))

        # now the network has converged

    # --- left off here (going to bed). read TODOs above please ---



# returns a measure of this networks performace growth rate
# (compare to some defined threshold to decide if the network has converge)
# returns true if network is deemed to have already converged
# based on the weighted average of the last `largBin` epochs' test results
# compared to that of the last `smallBin` epochs
#
# history: dict where epoch numbers are keys, and values is the
#   (float) performace of the network at that given point in time)
# threshHold: min performace (ratio) of two averages to consider
#   a network not converged yet.
# TODO: really should just be called weightedAverage()
def getConvergeRate(history, largeBin, smallBin, thresHold=0.005):
    if largeBin < smallBin:
        return False # dummy
    # take weighted average of the perforamce of the last `largeBin` epochs
    # and weighted average of the performance of the last `smallBin` epochs
    # (and compare them to decide if the network has stopped improving):
    keys = list(history.keys()).sort()

    if len(history) < largeBin: # ensure we have enough data in history
        return False # (why did you call this function so early?)

    sum_WeightedAverage = 0.0  # for computing how far back to look for weighted average
    decay = 1.0                # cur decay value
    decayUpdate = -0.2         # how much to change decay each time we step back on epoch?
    total_DecayWeight = 1.0    # sum of all weights used in taking the weighted average

    total = 0        # total for weighted average
    totalWeights = 0 # total of all the weights used (in weighted average)

    results = [] # results[0] = weighted average of last largeBin epochs
                 # results[1] = weighted average of last smallBin epochs
    bins = [largeBin, smallBin]
    for b in range(2):
        # get average of last b keys
        for i in range(0, b): # current offset from end of list
            prevKey = keys[-1 * (i+1)]
            curKey = keys[-i]

            # the bigger 'diff' is, the further we are from convergence
            # (this is even more true of the weighted average of diff that we are computing)
            diff = history[curKey] - history[prevKey]
            # (higher weights for more recent epochs)
            curWeight = (1.0 - i / (b))      # cur weight to use

            total += curWeight * diff
            toalWeights += curWeight
        results[i] = total / totalWeights # final weighted average
    ratio = (results[1] / results[0])
    #return ratio < (1.0 + threshold)
    return ratio

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


if __name__ == "__main__":
    main()

# note also: that in the GAN setup thing
# you could have another network learning to draw images that are NOT the concept
# (e.g. its cost funciton gives it the priority of seeking
# low prediction rates for the given concept (say the number '2')
# (further attempting to diversify the dataset for training) so trying to teach trick the `concept` detector into thinking it
#   (by having data examples where the expected output is (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0))
#   ### (maybe not) ### (if we told it to generate "not 2" we could even give all other outputs in the "ground truth" value a slight increase above 0

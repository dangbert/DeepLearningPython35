# Grammar Tree Experiments:

### Experiment Setup:
````python
# created 2 nets and trained them each to ~94% perfomance on mnist (5 epochs)
net0 = GrammarNet.Network([784, 70, 40, 15, 10], name="net0", backupDir="backups/grammarTree")
net1 = GrammarNet.Network([784, 55, 40, 21, 10], name="net1", backupDir="backups/grammarTree")
````

Then created a tmpNet with layers [55, 40, 15, 10] (formed by combining the two nets)
  (layer 0 being just an extra dummy layer to help with computing errors in layer 1 using existing `backprop()` code).

Then trained net1, while modifying its cost function to simulate its grammar layers outputs being fed into the tmpNet (in reality feeding its outputs from layer 1).

### Results:

The accuracy of net1 as it was trained immediately dropped to 18.6 after the first epoch of grammar training, and stayed relatively constant around 18-20% for the next 20 epochs.

Note that this error also came up in training:

> /home/dan/.dan/projects/DeepLearningPython35/mynet.py:291: RuntimeWarning: overflow encountered in exp return 1.0 / (1.0 + np.exp(-z))   


Training net1 and net2 in alternating epochs, net0 was at about 9-11% performance, and net1 at 79-82% (after 28 epochs each).

In a repeat of this experiment, both networks performed around 10% accuracy (after 25 epochs)...
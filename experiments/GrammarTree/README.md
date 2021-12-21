# Grammar Tree Experiments:

````python
# created 2 nets and trained them each to ~94% perfomance on mnist (5 epochs)
net0 = GrammarNet.Network([784, 70, 40, 15, 10], name="net0", backupDir="backups/grammarTree")
net1 = GrammarNet.Network([784, 55, 40, 21, 10], name="net1", backupDir="backups/grammarTree")
````

Then created a tmpNet with layers [55, 40, 15, 10] (formed by combining the two nets)
  (layer 0 being just an extra dummy layer to help with computing errors in layer 1 using existing `backprop()` code).

Then trained net1, while modifying its cost function to simulate its grammar layers outputs being fed into the tmpNet (in reality feeding its outputs from layer 1).

Results:

The accuracy of net1 as it was trained immediately dropped to 18 percent and oscillated between there and ~10% over 25 epochs (tending towards ~10%)

But this error at the start of training may have been an influence.

> /home/dan/.dan/projects/DeepLearningPython35/mynet.py:291: RuntimeWarning: overflow encountered in exp return 1.0 / (1.0 + np.exp(-z))   
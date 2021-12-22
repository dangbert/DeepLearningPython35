# Grammar Tree Experiments:

### Experiment Setup:
````python
# created 2 nets and trained them each to ~94% perfomance on mnist (5 epochs)
net0 = GrammarNet.Network([784, 70, 40, 15, 10], name="net0", backupDir="backups/grammarTree")
net1 = GrammarNet.Network([784, 55, 40, 21, 10], name="net1", backupDir="backups/grammarTree")
````

Then created a tmpNet by copying the net formed by the last 3 layers ([40, 15, 10]) of net0.

Then trained net1, with a modiified cost function to simulate its grammar layer (of size 40) outputs being fed into the tmpNet as well.
So the blame values in its grammar layer were modified to add a component of blame for the performance of the layers activations being fed into the tmpNet as input.  The tmpNet was never modified during the training.

### Results:

net1 was trained like this for 100 additional epochs. On the first additional epoch it had performace 94.12%, and reached ~96% accuracy after 35 total additional epochs.  After 105 total additional epochs it reached 96.45% percent.  The performance never dipped below 94% and was quite stable over time.


TODO: analyze performace now when the outputs of the grammer layer are fed into the tmpNet.
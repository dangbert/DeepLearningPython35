# Grammar Tree Experiments:

### Experiment Setup:
````python
# created 2 nets and trained them each to ~94% perfomance on mnist (5 epochs)
net0 = GrammarNet.Network([784, 70, 40, 15, 10], name="net0", backupDir="backups/grammarTree")
net1 = GrammarNet.Network([784, 55, 40, 21, 10], name="net1", backupDir="backups/grammarTree")
````

Then created a tmpNet by copying the net formed by the last 3 layers ([40, 21, 10]) of net1.

Then trained net0, with a modiified cost function to simulate its grammar layer (of size 40) outputs being fed into the tmpNet as well.
So the blame values in its grammar layer were modified to add a component of blame for the performance of the layers activations being fed into the tmpNet as input.  The tmpNet was never modified during the training.

### Results:

net0 was trained like this for ?? additional epochs. On the first additional epoch it had performace 2%, and reached ~96% accuracy after 35 total additional epochs.  After 105 total additional epochs it reached 96.45% percent.  The performance never dipped below 94% and was quite stable over time.


TODO: analyze performace now when the outputs of the grammer layer are fed into the tmpNet.

tmp0 performance is the performance of grammar subnet of net0, when the outputs of net1 are fed into it. This tests if the grammar layer works well in the opposite direction, even though it wasn't trained that way (yet).


epoch, tmp1 performace, net0 performance, tmp0 performace
epoch5 (initial), 4.9%, 94.12%
epoch6, 46.6%, 94.24%
epoch 15, 65.8%, 96.2%, 
epoch 21, 66.1%, 96.3%, 
epoch 22, 77.4%, 96.3% (big jump), 
epoch 24, 77.3%, 96.2%, 18.23%
epoch 40, 87.0%, 96.5%, 15.3%
epoch 50, 86.3%, 95.86%, 16.2%
epoch 98, 86.9%, 96.3%, 23.7%
epoch 153, 86.3%, 96.4%, 31.9%
epoch 221, 86.7%, 96.5%, 33.7%
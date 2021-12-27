# Grammar Tree Experiments:

Note: for all experiments, unless otherwise stated, the performance of a network on a given dataset is the percentage of dataset items for which the highest output activation of the network corresponded to the correct data label.

## Experiment2:
[link to code at time of experiment](https://github.com/dangbert/DeepLearningPython35/tree/c97c746c65880c004aa01b6c0ef235d1b5326cf9)

### Setup:
````python
# created 2 nets and trained them each to ~94% perfomance on mnist (5 epochs)
net0 = GrammarNet.Network([784, 70, 40, 15, 10], name="net0", backupDir="backups/grammarTree")
net1 = GrammarNet.Network([784, 55, 40, 21, 10], name="net1", backupDir="backups/grammarTree")
````

> Note: we define "grammar layer" here as they layer of size 40 (in either network).

Then created net tmp1 by copying the network formed by the last 3 layers ([40, 21, 10]) of net1.

Then trained net0, with a modified cost function to simulate its grammar layer (of size 40) outputs being fed into tmp1 as well.
So the blame values in its grammar layer were modified to add a component of blame for the performance of the layers activations being fed into tmp1 as input.  Note that tmp1 was never modified during the training.

### Results:

net0 was trained like this for 1090 additional epochs. 

The performance of tmp0 was also tracked; where tmp0 was the subnet of net0 (from the grammer layer to the final layer), with the activations of net1's grammar layer fed into it for evaluation purposes.
This tests if the grammar layer works well in the opposite direction, even though it wasn't trained that way (yet).  Meaning, are the activations of net1's grammar layer intelligible to net0's grammar layer?

#### Data:
See `experiment2_stats.png` and `experiment2_stats.pkl`

<img src="./archive/experiment2/experiment2_stats.png?raw=true" alt="main view" width="550">



Note: I wasn't tracking all metrics from the beginning, so see below the metrics I manually tracked at the start:

| epoch  |  tmp1   |  net0 |  tmp0 |
|--------|---------|-------|-------|
| 6      |  46.6%  | 94.2% |       |
| 15     | 65.8%   | 96.2% |       | 
| 21     | 66.1%   | 96.3% |       |
| 22     | 77.4%   | 96.3% |       |
| 24     | 77.3%   | 96.2% | 18.2% |
| 40     | 87.0%   | 96.5% | 15.3% |
| 50     | 86.3%   | 95.9% | 16.2% |
| 98     | 86.9%   | 96.3% | 23.7% |
| 153    | 86.3%   | 96.4% | 31.9% |
| 221    | 86.7%   | 96.5% | 33.7% |


## Experiment2b
[link to code at time of experiment](https://github.com/dangbert/DeepLearningPython35/tree/62b3fa7b3e75528013df2487dc9b05c3737b837f)

Exact same as experiment2, except interleaved training net0 and net1 from scratch, (using the final half of the opposite network as a tmp net for training).  Where tmp1 is the second half of net1, and tmp0 is the second half of net0.

<img src="./archive/experiment2b/stats.png?raw=true" alt="main view" width="550">

The final trained networks (net0 and net1) can each be split into 2 subnets, forming net0A and net0B (from net0 split at the grammary layer), and net1A and net1B (from net1).

4 unique, complete networks were formed from these subnetworks (2 of them being just net0 and net1 exactly).
These 4 networks were then evaluated on each item in the test dataset and allowed to "vote" on the correct result.  Results are as follows:


| net0        | net0A-net1B | net1        | net1A-net0B | pool vote   |
|-------------|-------------|-------------|-------------|-------------|
| 96.6%       |  96.4%      | 96.2%       | 95.8%       | 97.0%       |

Note: the pool vote when averaging each networks output resulted specifically in a 97.03% performance, and 97.06% when taking the median of each networks output.
# Deep Network Feedback Experiments:

Note: for all experiments, unless otherwise stated, the performance of a network on a given dataset is the percentage of dataset items for which the highest output activation of the network corresponded to the correct data label.

## Background
These experiments test the idea of a network that passes outputs from a deep layer back to its input layer.

So for a given "raw" input, the network can be ran for several iterations, and hopefully improve its performance over the iterations.

On the first iteration, we inject all zeros or random gaussian noise as a placeholder for the initial deep feedback.
    
Note to self: see my email [here](https://mail.google.com/mail/u/0/#inbox/KtbxLwgxChPRnrCsrZxDLJPCzHLxxlJgJq) with full initial thoughts on this idea.  This idea itself is inspired by how deep feedback in the brain's visual system feeds back to the earlier levels of the visual cortex.  "What we think we're seeing influences what we actually see."  [more info here](https://medicalxpress.com/news/2021-06-feedback-visual-cortex-perception.html).

## Experiment 1 (a):
[link to code at time of experiment](https://github.com/dangbert/DeepLearningPython35/tree/e304ac447f76ec9875a090e50a9d0f24016ee82c)


Two networks were trained, one as an instance of mynet.Network (as a baseline), and a second as an instance of my new Deep Feedback network:

````python
FEEDBACK_DIM = 15
sizes = [784, 70, 40, FEEDBACK_DIM, 10]
netControl = network.Network(sizes, name="netControl", backupDir=BACKUP_DIR)
# automatically handles resizing the input layer to make room for FEEDBACK_DIM additional inputs:
net0 = DNetwork.Network(sizes, name="net0", backupDir=BACKUP_DIR)
````

Both networks were then trained for 200 epochs on the mnist dataset.  net0 was trained such that each raw training input was "augmented", by appending FEEDBACK_DIM zeros to the end of it, getting the final output of the network, then running a second iteration where the previous output was appended to the raw input (augmenting it again).  On this second interation the network is actually trained.  It's essentially trained as though the augmented input was just a typical static input, so there are no real changes to the training algorithm beyond simply augmenting/expanding the dimensionality of the original input.

### Results:

No notable improvement at this time.

#### Data:
See `stats.pkl` in the `archive/experiment1` folder.  The stats/plot below capture the performance of both networks, sampled every 25 epochs (up to epoch 200).

<img src="./archive/experiment1/stats.png?raw=true" alt="main view" width="550">

Performance of net0 (epoch200) with varying total iterations (during test stage):

<img src="./archive/experiment1/stats_iterations.png?raw=true" alt="main view" width="550">
(remember the network was trained to optimize its performance on iteration 2).


## Experiment 1b:
[link to code at time of experiment](https://github.com/dangbert/DeepLearningPython35/tree/2d6e407ce02b9d4721d69bdc531b9b56fd3bd164)

Same as experiment 1a, except inject random gaussian noise as seed for first iteration of a given input:

````python
# seed with all zeros:
#prevFeedback = np.zeros((self.feedbackDim, 1))
# seed with random gaussian noise:
prevFeedback = np.random.normal(1, 0.5, (self.feedbackDim, 1))
````

Additionally, the network was trained using the 4th iteration of each input, rather than iteration 2 (as experiment1a did).  (note: removed "control" network from this experiment).


First 37 epochs:

<img src="./archive/experiment1b/stats_epoch0037.png?raw=true" alt="main view" width="550">
<img src="./archive/experiment1b/stats_iterations_epoch0037.png?raw=true" alt="main view" width="550">

1330 epochs:

<img src="./archive/experiment1b/stats.png?raw=true" alt="main view" width="550">

<img src="./archive/experiment1b/stats_iterations.png?raw=true" alt="main view" width="550">
<img src="./archive/experiment1b/stats_50_iterations.png?raw=true" alt="main view" width="550">

---
## Further Experiments/ideas:
* ~~Try doing a higher number of iterations (only 2 were used for this experiment)!!~~
* ~~Experiment with passing noise to augment the raw input on the first iteration.~~
* Experiment with combining this with the grammar tree experiment (i.e. passing feedback from pooled grammar tree layer(s) back to earlier layers in the networks)...
* Try passing feedback not to layer 0, but to a later layer (like layer 1).
* Consider if we could/should do some form of training on every pass, rather than just the final pass.

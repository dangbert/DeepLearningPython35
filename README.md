# DeepLearningPython35

## Purpose

This is a learning project for developing neural networks from scratch in python.  This is my work as I follow the [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/) book guide.

**General Goal: Make it through the book, then later consider greateer code deviations from the book (or start a new branch/repo)/**

* This repository is a fork of [MichalDanielDobrzanski/DeepLearningPython35](https://github.com/MichalDanielDobrzanski/DeepLearningPython35) which is itself a python 3.5 port of the book's provided [code repository](https://github.com/mnielsen/neural-networks-and-deep-learning).

## Overview

#### My Main Files:
* `test.py`:
  * Contains all three networks (network.py, network2.py, network3.py) from the book and it is the starting point to run (i.e. *train and evaluate*) them.
  * Contains examples of networks configurations with proper comments. I did that to relate with particular chapters from the book.
* `mynet.py`: TODO
* `learning-network.ipynb`: Jupyter Notebook for experimenting
* `bootyNet.py`: incomplete personal idea not directly related to this book

#### Provided Files (maybe slightly modified):
* `network.py`: 
* `network2.py`: 
* `network3.py`: 
* mnist files:
  * `mnist.pkl.gz`: mnist data set
  * `expand_mnist.py`
  * `mnist_average_darkness.py`
  * `mnist_loader.py`
  * `mnist_svm.py`

---
## Future Ideas:
* TODO: setup sphinx docs
* Generate adversarial images
* Implement GAN's?
* Revisit bootyNet.py and look over my [neural net ideas list](https://keep.google.com/u/0/#NOTE/14hIjnAyM_VcuLiRy_NhfbPioc1V45UAdlQHWkBRois_9T1yieWDTIz5UO2TdYuR7L3q0aahx)
* Do some cool experiments with visualizing neurons in my network:
  * https://distill.pub/2017/feature-visualization/
  * https://distill.pub/2020/circuits/zoom-in/
  * [manim library](https://github.com/3b1b/manim)
    * [Getting Started Tutorial](https://talkingphysics.wordpress.com/2019/01/08/getting-started-animating-with-manim-and-python-3-7/)

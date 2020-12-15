# DeepLearningPython35

## Purpose

This is a learning project for developing neural networks from scratch in python.  This is my work as I follow the [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/) book guide.

**General Goal: Make it through the book, then later consider greateer code deviations from the book (or start a new branch/repo)/**

* This repository is a fork of [MichalDanielDobrzanski/DeepLearningPython35](https://github.com/MichalDanielDobrzanski/DeepLearningPython35) which is itself a python 3.5 port of the book's provided [code repository](https://github.com/mnielsen/neural-networks-and-deep-learning).

## Overview

#### Provided Files:
> Maybe slightly modified from initial provided code in repo.

* `test.py`:
  * File for running (training/evaluating) all networks provided with the initial code repo  (see below).  (Note: I haven't made any substantial changes).
* `network.py`: Neural net implementation provided (with initial repo) for chapters [1](http://neuralnetworksanddeeplearning.com/chap1.html), [2](http://neuralnetworksanddeeplearning.com/chap2.html).
* `network2.py`: Neural net implementation provided for chapters [3](http://neuralnetworksanddeeplearning.com/chap3.html), [4](http://neuralnetworksanddeeplearning.com/chap4.html), [5](http://neuralnetworksanddeeplearning.com/chap5.html).
* `network3.py`: Neural net implementation provided for chapter [6](http://neuralnetworksanddeeplearning.com/chap6.html) (the last chapter).
* mnist files:
  * `mnist.pkl.gz`: mnist data set
  * `expand_mnist.py`
  * `mnist_average_darkness.py`
  * `mnist_loader.py`
  * `mnist_svm.py`
  
  
#### My Personal Files:
* `test1.py`: My test runner for `mynet.py`
* `mynet.py`: My neural net implementation based on network.py 
* `learning-network.ipynb`: Jupyter Notebook for experimenting
* `bootyNet.py`: incomplete personal idea not directly related to this book

## Build docs:
````bash
# setup virtualenv (if not done previsouly):
virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
cd docs
# build html docs:
make html
# build pdf docs:
sudo apt-install latexmk texlive-latex-recommended texlive-latex-extra
make pdflatex
````

---
## Future Ideas:
* Generate adversarial images
* Implement GAN's?
* Revisit bootyNet.py and look over my [neural net ideas list](https://keep.google.com/u/0/#NOTE/14hIjnAyM_VcuLiRy_NhfbPioc1V45UAdlQHWkBRois_9T1yieWDTIz5UO2TdYuR7L3q0aahx)
* Do some cool experiments with visualizing neurons in my network:
  * https://distill.pub/2017/feature-visualization/
  * https://distill.pub/2020/circuits/zoom-in/
  * [manim library](https://github.com/3b1b/manim)
    * [Getting Started Tutorial](https://talkingphysics.wordpress.com/2019/01/08/getting-started-animating-with-manim-and-python-3-7/)

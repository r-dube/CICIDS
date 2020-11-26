---
title: Reimplementing the neural network classifier with keras
---
We reimplement the neural classifier using keras to develop a feel for the difference between keras and pytorch.

### Motivation
As of this writing, keras (with tensorflow) rivals pytorch in popularity as a neural network implementation framework. We anticipate that some neural network tasks may be more convenient using one framework than the other. We reimplement the fully-connected neural network classifier from ([^colab2]) using keras to increase our familarity with keras ([^colab8]).

### Similarities
Translating the pytorch neural network that previously produced good results to keras is straightforward. Most of the information needed for this translation is covered in online tutorials ([^lizard1]). For fully-connected neural networks, keras appears to have all the equivalent constructs to those in pytorch.

### Differences
In the pytorch implementation, we had to write a small amount of code to carry out training. Writing this code required some familiarities with pytorch's internals and some faith that template code copied over from tutorials would work even if one did not fully understand pytorch's internals. Keras hides the details of training and backward propagation better than pytorch. As a result, creating a neural network with keras feels a little more intuitive and clean.

We did run into one problem with the keras implementation that took a bit of time to diagnose and rectify. In the pytorch implementation, we used cross-entropy loss with linear activation at the output layer. It turns out that the equivalent implentation in keras requires sparse-categorical-cross-entropy loss with softmax activation at the output layer. Pytorch tacitly employs softmax inside its cross-entropy loss function whereas keras does not. We only discovered this difference after the keras implementation failed to produce results close to those seen with the pytorch implementation.

### Future experimentation
Given the experience documented above, we expect to use keras for future experimentation.

### References
[^colab2]: [Logistic, neural networks, KNN code for Colab](https://github.com/r-dube/CICIDS/blob/main/ids_classifiers.ipynb)
[^colab8]: [NN reimplementation with Keras and Tensorflow](https://github.com/r-dube/CICIDS/blob/main/ids_heartbleed.ipynb)
[^lizard1]: [Keras - Python Deep Learning Neural Network API](https://deeplizard.com/learn/video/RznKVRTFkBY)

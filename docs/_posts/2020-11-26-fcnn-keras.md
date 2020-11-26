---
title: Reimplementing the neural network classifier with Keras
---
We reimplement the neural classifier using Keras to develop a feel for the difference between Keras and PyTorch.

### Motivation
As of this writing, Keras (with TensorFlow) rivals PyTorch in popularity as a neural network implementation framework. We anticipate that some neural network tasks may be more convenient using one framework than the other. We reimplement the fully-connected neural network classifier from ([^colab2]) using Keras to increase our familiarity with Keras ([^colab8]).

### Similarities
Translating the PyTorch neural network that previously produced good results to Keras is straightforward. Most of the information needed for this translation is covered in online tutorials ([^lizard1]). For fully-connected neural networks, Keras appears to have all the equivalent constructs to those in PyTorch.

### Differences
In the PyTorch implementation, we had to write a small amount of code to carry out training. Writing this code required some familiarities with PyTorch's internals and some faith that template code copied over from tutorials would work even if one did not fully understand PyTorch's internals. Keras hides the details of training and backward propagation better than PyTorch. As a result, creating a neural network with Keras feels a little more intuitive and clean.

We did run into one problem with the Keras implementation that took a bit of time to diagnose and rectify. In the PyTorch implementation, we used cross-entropy loss with linear activation at the output layer. It turns out that the equivalent implementation in Keras requires sparse-categorical-cross-entropy loss with softmax activation at the output layer. Pytorch tacitly employs softmax inside its cross-entropy loss function, whereas Keras does not. We only discovered this difference after the Keras implementation failed to produce results close to those seen with the PyTorch implementation.

### Future experimentation
Given the experience documented above, we expect to use Keras for future experimentation.

### References
[^colab2]: [Logistic, neural networks, KNN code for Colab](https://github.com/r-dube/CICIDS/blob/main/ids_classifiers.ipynb)
[^colab8]: [NN reimplementation with Keras and TensorFlow](https://github.com/r-dube/CICIDS/blob/main/ids_keras_tf.ipynb)
[^lizard1]: [Keras - Python Deep Learning Neural Network API](https://deeplizard.com/learn/video/RznKVRTFkBY)

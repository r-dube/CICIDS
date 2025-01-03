---
title: Using neural networks
---
We attempt to beat the baseline classification accuracy of logistic regression with a neural network-based classifier.

### Network design
We use a simple fully-connected neural network with one hidden layer. The hidden layer is the same size as the input layer. In turn, the input layer has one node for each feature retained in the processed data ([^data1]). The number of output nodes is the same as the number of classes. Cross-entropy loss is used as the loss function to guide backpropagation ([^colab2]).

The network is coded using Python's Pytorch.

### Standardizing data
As with logistic regression, when the processed data is used as is, convergence is slow. Also, classification results are poor when training is conducted for a small number (10 - 30) of epochs. Standardizing the data speeds up convergence and improves classification accuracy.

### Classification accuracy
Experimenting with various hyper-parameters, we get 97% accuracy with 10 training epochs. The classification results are notably better than those produced by logistic regression.

Other hyper-parameters and additional training get the neural network beyond 98% accuracy. However, the process of systematic experimentation requires extra code and several more training runs.


### References
[^data1]: [Processed data](https://github.com/r-dube/CICIDS/blob/main/MachineLearningCVE/processed/bal-cicids2017.csv)
[^colab2]: [Logistic, neural networks, KNN code for Colab](https://github.com/r-dube/CICIDS/blob/main/ids_classifiers.ipynb)

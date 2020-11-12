---
mathjax: true
title: Measuring classification performance
---
We use two measures for classification performance in this project: accuracy and F1-score.

### Accuracy
Multi-class accuracy is the primary measure of performance used in this project. We generate and record both multi-class accuracy and two-class accuracy. The former metric measures classification correctness across the "BENIGN" and the 12 attack classes. The latter metric lumps all the attack classes into one ([^colab2]).

### F1-score
We also record the F1-score for all experiments. The F1-score is a composite of the true positives (TP), false positives (FP), and false negatives (FN): $F1_{score} = \frac {TP} {TP + 0.5(FP + FN)}$. Such a score can be a better measure of performance than accuracy when we want to explicitly account for classification errors.

Note that both multi-class and two-class scores are generated.

We acknowledge that false negatives are more harmful in information security than false positives, but both are given equal weight in the F1-score. We may consider changing the relative weights of the two types of errors in future experiments.

### Performance on the validation set
We use the training set for training and the validation set for experimenting with classification hyper-parameters. 

All classification performance thus far has been reported on the validation set. The test set has been left untouched for a final comparison across classification techniques (sometime in the future).

### Confusion matrix
We generate a confusion matrix - that records the pairwise classification errors - on the validation set in some of the experiments. The confusion matrix gives us a sense of the classes that perform the worst with a classification technique. Currently, the confusion matrix is used as a sanity check rather than as a performance measure.

### References
[^colab2]: [Logistic, neural networks, KNN code for Colab](https://github.com/r-dube/CICIDS/blob/main/cicids_classifiers.ipynb)

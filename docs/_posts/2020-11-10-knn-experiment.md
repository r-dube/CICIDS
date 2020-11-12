---
title: Experimenting with KNN
---
We attempt classification using K-nearest-neighbors (KNN) to increase the diversity of techniques used with the dataset.

### Standardized data and a small number of neighbors
Having learned from logistic regression and neural networks, we only use standardized data for the KNN experiment. We also choose a small number of neighbors - 3 - to obtain results quickly. Finally, we leave the distance measure used by the KNN algorithm as the default (euclidean distance).

### Classification accuracy
We find that the KNN classifier returns an accuracy greater than 99%. The accuracy is surprising given the default settings used, fast convergence of the classifier, and the small amount of effort expended in setting up the classifier.

### Further investigation
We wonder if the KNN results are due to the specific type of processing applied to the data ([^data1]) or a lucky choice of parameters. KNN's classification performance versus logistic regression and the neural network-based classifier motivate further investigation into the KNN's training and classification characteristics.


### References
[^data1]: [Processed data](https://github.com/r-dube/CICIDS/blob/main/MachineLearningCVE/processed/bal-cicids2017.csv)
[^colab2]: [Logistic, neural networks, KNN code for Colab](https://github.com/r-dube/CICIDS/blob/main/cicids_classifiers.ipynb)

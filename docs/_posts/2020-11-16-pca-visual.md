---
title: Visualizing principal component data
---
We use principal component analysis (PCA) to extract components and attempt visualizing the data.

### Visualizing multi-dimensional data
Visualizing multi-dimensional data is difficult even when the number of samples in the data set is reduced. The (reduced) attack dataset ([^data2]) still contains almost all of the features in the original (raw) data set.

### Using PCA to reduce dimensions
We use Scikit-learn's PCA module to extract a handful (six) of principal components from the data ([^colab5]). Our hope is that plotting labels against a few pairs of principal components might reveal some pattern that improves our understanding of the data. 

Of course, PCA transforms the original features making interpretation of the axes in plots more difficult. Regardless an experiment appears worth the effort.

### Interpreting pair-wise plots
We plot the labeled data against the first six principal components. 

The first chart below has the first principal component on the x-axis and the second principal component on the y-axis. The label is mapped to a color. The second and third charts are similar but apply to principal components three, four and five, six respectively,

We see that while there is some pattern to the data, plotting the labels against first six components, two at a time, is not enough to demonstrate the margin boundary between the various attack classes.

Principal components one, two
![Principal components one, two](/CICIDS/assets/images/2020-11-16-pca-1.png)

Principal components three, four
![Principal components three, four](/CICIDS/assets/images/2020-11-16-pca-2.png)

Principal components five, six
![Principal components five, six](/CICIDS/assets/images/2020-11-16-pca-3.png)

### References
[^data2]: [Reduced attack data](https://github.com/r-dube/CICIDS/blob/main/MachineLearningCVE/processed/small-cicids2017.csv)
[^colab5]: [PCA experimentation on Colab](https://github.com/r-dube/CICIDS/blob/main/ids_pca.ipynb)

---
title: Developing a baseline
---
We use logistic regression as the first classification technique on the processed data to develop a baseline for classification results.

### Difficulty converging
On running logistic regression directly on the processed data ([^data1],[^colab1]), we find that the classification results are poor and logistic regression has difficulty converging. Python Scikit-learn's logistic regression parameter finding algorithm hits the default limit for the maximum number of (Scikit-learn internal) iterations even as classification accuracy languishes well below 90%.

### Standardizing data
On standardizing the data, we find that classification accuracy increases to about 92% ([^colab2]), even though Scikit-learn complains about hitting the default maximum iteration limit.

### Maximum iterations
To prevent Scikit-learn from complaining, we tried increasing the maximum number of iterations from 100 to 10000. The higher limit eliminates the complaint but runs the algorithm for a long time while topping out at 93% accuracy.

### Baseline accuracy
Given the experience above, we choose 92% as the baseline accuracy that other classification techniques would have to beat.


### References
[^data1]: [Processed data](https://github.com/r-dube/CICIDS/blob/main/MachineLearningCVE/processed/bal-cicids2017.csv)
[^colab1]: [Data processing code on Colab](https://github.com/r-dube/CICIDS/blob/main/cicids_data.ipynb)
[^colab2]: [Logistic, neural networks, KNN code on Colab](https://github.com/r-dube/CICIDS/blob/main/cicids_classifiers.ipynb)

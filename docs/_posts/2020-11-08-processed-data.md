---
title: On processed data
---
We process the raw CICIDS2017 data to get into a form that is usable by machine learning algorithms.

### Combining
The raw data consists of 8 comma-separated-values (CSV) files that total up to 283K rows (864MB). As a first step, we combine all the CSV files into one ([^colab1],[^scripts1]).

### Cleaning
Subsequently, we clean the data. 

The raw data is, in fact, relatively clean to begin with, likely because it was machine-generated. However, the data contains non-numeric values for "features" corresponding to "flow bytes per second" and "flow packets per second" ([^notes1]). We assume that the non-numeric values are due to a coding error (dating back to the creation of the dataset) and drop these features from the cleaned data. 

We might have to revisit the decision to drop the two features mentioned above at a future date.

### Balancing
An overwhelming majority of the data is from the "BENIGN" class ([^notes2]). While some attack classes such as "DDOS" have a healthy representation, other attack classes such as "Web Attack SQL Injection" are barely represented in the data. 

We resample the "BENIGN" class and the attack classes such that the "BENIGN" class has 40,000 entries, and each of the 12 attack classes has 8,000 entries.

### Other processing
To make the features and labels easier to reference, we change the label for each row from a string (example "BENIGN") to a number (example 0). Similarly, we change the feature labels from long strings (example "Destination Port" and "Label") to shorter strings (example "X1" and "YY").

### Processed data
We end up with 76 features (X1 - X14, X17 - X78), accounting for the two features dropped above (corresponding to X15, X16). We also have 13 numeric labels (0 - 12). The final processed file has 136,00 entries ([^data1]).


### References
[^notes1]: [Feature descriptions](https://github.com/r-dube/CICIDS/blob/main/notes/cicflowmeter-2020-ReadMe.txt)
[^notes2]: [Lab notes](https://github.com/r-dube/CICIDS/blob/main/notes/lab-notes.txt)
[^colab1]: [Data processing code on Colab](https://github.com/r-dube/CICIDS/blob/main/cicids_data.ipynb)
[^scripts1]: [Data processing script for local machine](https://github.com/r-dube/CICIDS/blob/main/scripts/ids_utils.py)
[^data1]: [Processed data](https://github.com/r-dube/CICIDS/blob/main/MachineLearningCVE/processed/bal-cicids2017.csv)

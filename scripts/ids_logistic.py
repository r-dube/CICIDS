#!/usr/bin/env python3

# Load the top modules that are used in multiple places
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler

from ids_utils import *

def ids_logistic():
    """
    Classify processed data set stored as csv file using logistic regression
    Print: accuracy, confusion matrix, f1 score on the validation data set
    Input:
        None    
    Returns:
        None
    """

    df = ids_load_df_from_csv (outdir, balanced_data)
    X_train, X_val, X_test, y_train, y_val, y_test = ids_split(df)

    # max_iter could be set to a large value (10000) to prevent 
    # LogisticRegression() from complaining that # it is not coverging
    logreg = LogisticRegression(max_iter=100)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_val)
    
    ids_metrics(y_val, y_pred)

ids_logistic()

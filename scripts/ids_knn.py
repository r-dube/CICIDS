#!/usr/bin/env python3

from ids_utils import *
from sklearn.neighbors import KNeighborsClassifier

def ids_knn():
    """
    Classify processed data set stored as csv file using KNN
    Print: accuracy, confusion matrix, f1 score on the validation data set
    Input:
        None    
    Returns:
        None
    """

    df = ids_load_df_from_csv (outdir, balanced_data)
    X_train, X_val, X_test, y_train, y_val, y_test = ids_split(df)

    # max_iter set to a large value to prevent LogisticRegression() from complaining that
    # it is not coverging
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_val)
    
    ids_metrics(y_val, y_pred)

ids_knn()

#!/usr/bin/env python3

"""
Data Cleaning and Utility functions for CICIDS 2017 data
"""

# Load the top modules that are used in multiple places
import numpy as np
import pandas as pd

# Some global variables to drive the script
# The indir should match the location of the data
# The outdir should be the desired location of the output
indir = '/Users/rdube/Repos/CICIDS/MachineLearningCVE/raw/'
outdir = '/Users/rdube/Repos/CICIDS/MachineLearningCVE/processed/'
combined_data='cicids2017.csv'
balanced_data='bal-cicids2017.csv'
small_data='small-cicids2017.csv'

# Uncomment for testing the process of combining data
# Balancing is not tested
# indir = '/Users/rdube/Repos/CICIDS/MachineLearningCVE/test/'
# outdir = '/Users/rdube/Repos/CICIDS/MachineLearningCVE/processed/'
# combined_data='test-cicids2017.csv'
# small_data='test-small-cicids2017.csv'

# Column name mapping from original data to compact form
# All the X** are features and the YY is the label
feature_map = {
 ' Destination Port' : 'X1',
 ' Flow Duration' : 'X2', 
 ' Total Fwd Packets' : 'X3', 
 ' Total Backward Packets' : 'X4', 
 'Total Length of Fwd Packets' : 'X5', 
 ' Total Length of Bwd Packets' : 'X6', 
 ' Fwd Packet Length Max' : 'X7', 
 ' Fwd Packet Length Min' : 'X8', 
 ' Fwd Packet Length Mean' : 'X9', 
 ' Fwd Packet Length Std' : 'X10', 
 'Bwd Packet Length Max' : 'X11', 
 ' Bwd Packet Length Min' : 'X12', 
 ' Bwd Packet Length Mean' : 'X13', 
 ' Bwd Packet Length Std' : 'X14', 
 'Flow Bytes/s' : 'X15', 
 ' Flow Packets/s' : 'X16', 
 ' Flow IAT Mean' : 'X17', 
 ' Flow IAT Std' : 'X18', 
 ' Flow IAT Max' : 'X19', 
 ' Flow IAT Min' : 'X20', 
 'Fwd IAT Total' : 'X21', 
 ' Fwd IAT Mean' : 'X22', 
 ' Fwd IAT Std' : 'X23', 
 ' Fwd IAT Max' : 'X24', 
 ' Fwd IAT Min' : 'X25', 
 'Bwd IAT Total' : 'X26', 
 ' Bwd IAT Mean' : 'X27', 
 ' Bwd IAT Std' : 'X28', 
 ' Bwd IAT Max' : 'X29', 
 ' Bwd IAT Min' : 'X30', 
 'Fwd PSH Flags' : 'X31', 
 ' Bwd PSH Flags' : 'X32', 
 ' Fwd URG Flags' : 'X33', 
 ' Bwd URG Flags' : 'X34', 
 ' Fwd Header Length' : 'X35', 
 ' Bwd Header Length' : 'X36', 
 'Fwd Packets/s' : 'X37', 
 ' Bwd Packets/s' : 'X38', 
 ' Min Packet Length' : 'X39', 
 ' Max Packet Length' : 'X40', 
 ' Packet Length Mean' : 'X41', 
 ' Packet Length Std' : 'X42', 
 ' Packet Length Variance' : 'X43', 
 'FIN Flag Count' : 'X44', 
 ' SYN Flag Count' : 'X45', 
 ' RST Flag Count' : 'X46', 
 ' PSH Flag Count' : 'X47', 
 ' ACK Flag Count' : 'X48', 
 ' URG Flag Count' : 'X49', 
 ' CWE Flag Count' : 'X50', 
 ' ECE Flag Count' : 'X51', 
 ' Down/Up Ratio' : 'X52', 
 ' Average Packet Size' : 'X53', 
 ' Avg Fwd Segment Size' : 'X54', 
 ' Avg Bwd Segment Size' : 'X55', 
 ' Fwd Header Length.1' : 'X56', 
 'Fwd Avg Bytes/Bulk' : 'X57', 
 ' Fwd Avg Packets/Bulk' : 'X58', 
 ' Fwd Avg Bulk Rate' : 'X59', 
 ' Bwd Avg Bytes/Bulk' : 'X60', 
 ' Bwd Avg Packets/Bulk' : 'X61', 
 'Bwd Avg Bulk Rate' : 'X62', 
 'Subflow Fwd Packets' : 'X63', 
 ' Subflow Fwd Bytes' : 'X64', 
 ' Subflow Bwd Packets' : 'X65', 
 ' Subflow Bwd Bytes' : 'X66', 
 'Init_Win_bytes_forward' : 'X67', 
 ' Init_Win_bytes_backward' : 'X68', 
 ' act_data_pkt_fwd' : 'X69', 
 ' min_seg_size_forward' : 'X70', 
 'Active Mean' : 'X71', 
 ' Active Std' : 'X72', 
 ' Active Max' : 'X73', 
 ' Active Min' : 'X74', 
 'Idle Mean' : 'X75', 
 ' Idle Std' : 'X76', 
 ' Idle Max' : 'X77', 
 ' Idle Min' : 'X78', 
 ' Label': 'YY'
}

# label names (YY) in the data and their
# mapping to numerical values
label_map = {
 'BENIGN' : 0,
 'FTP-Patator' : 1,
 'SSH-Patator' : 2,
 'DoS slowloris' : 3,
 'DoS Slowhttptest': 4,
 'DoS Hulk' : 5,
 'DoS GoldenEye' : 6,
 'Heartbleed' : 7,
 'Web Attack � Brute Force' : 8,
 'Web Attack � XSS' : 8,
 'Web Attack � Sql Injection' : 8,
 'Infiltration' : 9,
 'Bot' : 10,
 'PortScan' : 11,
 'DDoS' : 12,
}

num_ids_features = 76
num_ids_classes = 13
ids_classes = [ 'BENIGN', 'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye', 'Heartbleed', 'Brute Force', 'XSS', 'Sql Injection', 'Infiltration', 'Bot', 'PortScan', 'DDoS',]

def ids_combine():
    """
    Combine all csv files to produce a single csv file 
    Input:
        None
    Returns:
        None

    """

    import os
    import glob
    os.chdir(indir)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    # combine all files in the list
    df = pd.concat([pd.read_csv(f) for f in all_filenames ])

    # Drop columns 14 and 15 that have Nan and Infinity in them
    df.rename(columns = feature_map, inplace=True)
    df.drop(columns=['X15', 'X16'], inplace=True)

    # Convert string labels to numeric
    df['YY'].replace(label_map, inplace=True)

    # export to csv
    df.to_csv(outdir + combined_data, index=False)

def ids_balance():
    """
    Balance dataset using a heuristic
    Input:
        None
    Returns:
        None
    """

    from sklearn.utils import resample
    n = 8000

    df = pd.read_csv(outdir + combined_data, delimiter=',')
    df0 = df[df.YY == 0]
    df1 = df[df.YY == 1]
    df2 = df[df.YY == 2]
    df3 = df[df.YY == 3]
    df4 = df[df.YY == 4]
    df5 = df[df.YY == 5]
    df6 = df[df.YY == 6]
    df7 = df[df.YY == 7]
    df8 = df[df.YY == 8]
    df9 = df[df.YY == 9]
    df10 = df[df.YY == 10]
    df11 = df[df.YY == 11]
    df12 = df[df.YY == 12]
    
    df0 = resample(df0, replace=False, n_samples=5*n, random_state=123)
    df1 = resample(df1, replace=True, n_samples=n, random_state=123)
    df2 = resample(df2, replace=True, n_samples=n, random_state=123)
    df3 = resample(df3, replace=True, n_samples=n, random_state=123)
    df4 = resample(df4, replace=True, n_samples=n, random_state=123)
    df5 = resample(df5, replace=False, n_samples=n, random_state=123)
    df6 = resample(df6, replace=False, n_samples=n, random_state=123)
    df7 = resample(df7, replace=True, n_samples=n, random_state=123)
    df8 = resample(df8, replace=True, n_samples=n, random_state=123)
    df9 = resample(df9, replace=True, n_samples=n, random_state=123)
    df10 = resample(df10, replace=True, n_samples=n, random_state=123)
    df11 = resample(df11, replace=False, n_samples=n, random_state=123)
    df12 = resample(df12, replace=False, n_samples=n, random_state=123)

    df_sampled = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12])

    print (df_sampled.YY.value_counts())

    # export to csv
    df_sampled.to_csv(outdir + balanced_data, index=False)

def ids_small_classes():
    """
    Extract all rows for the smallest 9 of 12 attack classes
    Input:
        None
    Returns:
        None

    """
    df = pd.read_csv(outdir + combined_data, delimiter=',')

    df = df[(df.YY > 0) & (df.YY < 11) & (df.YY != 5)]

    print (df.YY.value_counts())

    # export to csv
    df.to_csv(outdir + small_data, index=False)

def ids_load_df_from_csv(dir, file):
    """
    Load dataframe from csv file
    Input:
        dir: directory for csv file
        file: csv file
    Returns:
        Pandas dataframe corresponding to processed and saved csv file
    """

    df = pd.read_csv(dir + file)

    print ("load Dataframe shape", df.shape)

    return df

def ids_split(df):
    """
    Input:
        Dataframe that has columns of covariates followed by a column of labels
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays
    """

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    numcols = len(df.columns)
    print("df.shape", df.shape)

    X = df.iloc[:, 0:numcols-1]
    y = df.loc[:, 'YY']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    print ("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
    print ("X_val.shape", X_val.shape, "y_val.shape", y_val.shape)
    print ("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values

    return X_train, X_val, X_test, y_train, y_val, y_test

def ids_shuffle_in_unison(a, b):
    np.random.seed(42)
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def ids_perturb(x_train, y_train):
    perturbation = 0.0001

    add_train = x_train.copy()
    add_train = add_train * (1 + perturbation)

    sub_train = x_train.copy()
    sub_train = sub_train * (1 - perturbation)

    X_train = np.vstack((x_train, add_train, sub_train))
    y_train = np.concatenate((y_train, y_train, y_train))

    # removing shuffling as logistic doesn't need it, fcnn does its own
    # X_train, y_train = ids_shuffle_in_unison(X_train, y_train)

    return X_train, y_train

def ids_accuracy (y_actual, y_pred):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    # modify labels to get results for two class classification
    y_actual_2 = (y_actual > 0).astype(int)
    y_pred_2 = (y_pred > 0).astype(int)

    acc = accuracy_score (y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred, average='macro')
    acc_2 = accuracy_score (y_actual_2, y_pred_2)
    f1_2 = f1_score(y_actual_2, y_pred_2)
    
    return acc, f1, acc_2, f1_2
    

def ids_metrics(y_actual, y_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix (y_actual, y_pred)
    print (cm)

    acc, f1, acc_2, f1_2 = ids_accuracy (y_actual, y_pred)
    print('Classifier accuracy : {:.4f}'.format(acc), 'F1 score: {:.4f}'.format(f1))
    print('Two class classifier accuracy : {:.4f}'.format(acc_2), 'F1 score: {:.4f}'.format(f1_2))

def ids_check_version():
    """ Prints Python version in use """
    import sys
    print (sys.version)


# def main():
# ids_check_version()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# uncomment to clean and combine data files
# ids_combine()

# uncomment to create a class-balanaced version of the data
# only works for raw (not test) data
# ids_balance ()

# uncomment to create a ten-class verstion of the data
# ids_small_classes()

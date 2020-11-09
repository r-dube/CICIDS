#!/usr/bin/env python3

''' 
Fully connected three layer neural network using PyTorch
With utility functions based on deeplizard tutorial
https://deeplizard.com/learn/video/v5cngxo4mIg
'''

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader
from IPython.display import display, clear_output
import pandas as pd
import time

from itertools import product
from collections import namedtuple
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from ids_utils import *

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs
    
class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
    
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
    
        self.network = None

    def begin_run(self, run, network, loss_fn, train_inputs, train_targets, X_val, y_val):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
    
        self.network = network
        self.loss_fn = loss_fn
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.X_val = X_val
        self.y_val = y_val

    def end_run(self):
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        with torch.no_grad():
            loss = (self.loss_fn(self.network(self.train_inputs), self.train_targets)).item()

            val_inputs = torch.from_numpy(self.X_val).float()
            val_preds = self.network(val_inputs)
            y_pred = val_preds.argmax(dim=1)
            accuracy = accuracy_score (self.y_val, y_pred)
    
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
    
def ids_nn():
    params = OrderedDict(
        lr = [.008]
        ,batch_size = [256]
        ,num_epochs = [10]
        ,step_size = [5]
        ,gamma = [0.50]
    )

    rm = RunManager()

    df = ids_load_df_from_csv (outdir, balanced_data)
    X_train, X_val, X_test, y_train, y_val, y_test = ids_split(df)
    train_inputs = torch.from_numpy(X_train).float()
    train_targets = torch.from_numpy(y_train).long()
    train_ds = TensorDataset(train_inputs, train_targets)

    # Run for each combination of params
    for run in RunBuilder.get_runs(params):
        torch.manual_seed(42)
        print (run)

        network = nn.Sequential(
            nn.Linear(num_ids_features, num_ids_features)
            ,nn.ReLU()
            ,nn.Linear(num_ids_features, num_ids_classes)
        )

        train_dl = DataLoader(train_ds, run.batch_size, shuffle=True)
        opt = torch.optim.Adam(network.parameters(), run.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, run.step_size, run.gamma)
        loss_fn = F.cross_entropy

        rm.begin_run(run, network, loss_fn, train_inputs, train_targets, X_val, y_val) 
        # Training loop
        for epoch in range(run.num_epochs):
            rm.begin_epoch()

            for xb,yb in train_dl:
                pred = network(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()

            rm.end_epoch()
            sch.step() 
        rm.end_run()

    print(pd.DataFrame.from_dict(rm.run_data))

    val_inputs = torch.from_numpy(X_val).float()
    val_pred = network(val_inputs)

    # Since the model returns values for all num_ids_classes
    # The ids_class with the maximim value is picked as the label
    val_pred = val_pred.argmax(dim=1)

    # A numpy array is needed to evaluate metrics
    y_pred = val_pred.detach().to('cpu').numpy()
    ids_metrics(y_val, y_pred)

# main()
ids_nn()

# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2023-08-25 13:46:44
# @Last Modified by:   Your name
# @Last Modified time: 2023-08-25 14:14:42
#!/usr/bin/env python

import mne
from os.path import join as opj
import torch
from sklearn import datasets, linear_model, svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from braindecode.models import ShallowFBCSPNet,Deep4Net, EEGNetv4
from braindecode import EEGClassifier, EEGRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler, ProgressBar, EpochScoring
from train_script_between_part import trainingDL_between, training_nested_cv_between
from train_script_within_part import training_nested_cv_within, trainingDL_within
import torch.nn as nn
import pandas as pd
from sklearn.pipeline import make_pipeline
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
import json
import os
import joblib
from sklearn.linear_model import ElasticNet
import sys
from braindecode.preprocessing import exponential_moving_standardize
#from pyriemann.classification import MDM, TSclassifier
#from pyriemann.estimation import Covariances, Shrinkage

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set kind of Cross validation and task to perform 
#part = 'between' # 'between' or 'within' participant
#task = 'classification' # 'classification' or 'regression'
#dl = True # Whether to use a deep learning or standard ML model

#____________________________________________________________________________
# Application of cross validation for different models
# Load data


# Get the current working directory
current_directory = os.getcwd()

# Check if the keyword "lustre" is present in the current directory
cuda = "lustre" in current_directory #this is changed to false if no GPU
cc = "lustre" in current_directory

# If cuda is True and a GPU is available, set up GPU acceleration in PyTorch
# And set bidsroot according to device 
if cuda:
    model_name = sys.argv[1]
    part = sys.argv[2]
    bsize = 16  
    target = sys.argv[3]
    search_params = sys.argv[4]
    if torch.cuda.is_available():
        device = torch.device('cuda')  # PyTorch will use the default GPU
        torch.backends.cudnn.benchmark = True
    else:
        cuda = False
        device = torch.device('cpu')

    bidsroot = '/lustre04/scratch/mabus103/normalized_data/normalized_epo.fif'
    log_dir=f'/lustre04/scratch/mabus103/logs'
    model_dir=f'/lustre04/scratch/mabus103/models/'
    #log_dir=f'/lustre04/scratch/mabus103/ML_for_Pain_Prediction/logs'
elif 'media/mp' in current_directory: #MP's local machine
    model_name = "shallowFBCSPNetClassification"
    part = "between"
    target = "3_classes"
    search_params = False
    bsize = 16
    device = torch.device('cuda')
    bidsroot = 'data/cleaned_epo.fif'
    log_dir= 'logs'

elif "mplab" in current_directory:
    model_name = "shallowFBCSPNetClassification" #set the model to use. also determines dl and kind of task
    part = 'between' # 'between' or 'within' participant
    target = "3_classes"
    search_params = False
    bsize = 16
    device = torch.device('cpu')  # Use CPU if GPU is not available or cuda is False
    #bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/cleaned_epo.fif'
    bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
    log_dir='/home/mplab/Desktop/Mathilda/Project/code/ML_for_Pain_Prediction/logs'
    model_dir='/home/mplab/Desktop/Mathilda/Project/code/ML_for_Pain_Prediction/models'
else:
    model_name = "SGD" #set the model to use. also determines dl and kind of task
    part = 'between'# 'between' or 'within' participant
    target = "intensity"
    search_params = False
    bsize = 16
    device = torch.device('cpu')  # Use CPU if GPU is not available or cuda is False
    bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
    log_dir='/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs'
    model_dir='/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/models/'

data_path = opj(bidsroot)
# Load epochs oject, is already normalized and some epochs removed on compute canada
epochs = mne.read_epochs(data_path, preload=True)

# Normalize X
# If on compute canada:
if cc:
    # load already normalized X
    loaded_data = np.load('normalized_X.npz')  # For .npz format
    X = loaded_data['X']

    # Only use thermal for regression
    if target == "intensity" or target == "rating":
        # Load the metadata information from the original epochs object
        metadata_df = epochs.metadata

        # Find the indices where the task is "thermal" or "thermalrate"
        thermal_indices = np.where(metadata_df["task"].isin(["thermal", "thermalrate"]))[0]

        # Filter X based on the thermal indices
        X = X[thermal_indices]

        # Filter epochs based on the thermal indices
        epochs = epochs[thermal_indices]


else:  
    #remove epochs above threshold
    threshold = 20

    # Get the metadata DataFrame from the Epochs object
    metadata_df = epochs.metadata

    # Get indices of epochs that meet the threshold
    selected_indices = np.where(metadata_df["diff_intensity"] <= abs(threshold))[0]

    # Filter out epochs based on the diff_intensity threshold
    epochs = epochs[selected_indices]

    # Print the initial and final number of epochs
    print("Number of epochs before removal:", len(metadata_df))
    print("Number of epochs after removal:", len(epochs))

    if target == "intensity" or target == "rating":
        #only use thermal task for pain intensity
        selected_tasks = ["thermal", "thermalrate"]
        epochs = epochs[epochs.metadata["task"].isin(selected_tasks)]
        X = epochs.get_data()
    else:
        X = epochs.get_data()

    X = X*1e6 # Convert from V to uV  
    for epo in tqdm(range(X.shape[0]), desc='Normalizing data'): # Loop epochs
        X[epo, :, :] = exponential_moving_standardize(X[epo, :, :], factor_new=0.001, init_block_size=None) # Normalize the data

# Define the groups (participants) to avoid splitting them across train and test
groups = epochs.metadata["participant_id"].values


#____________________________________________________________________
# Classification

n_chans = len(epochs.info['ch_names'])
input_window_samples=X.shape[2]
if target == "3_classes" or target =="5_classes":
    n_classes_clas=int(target[0])


#__________________________________________________________________
# Models

# Choose parameters for nested CV
if model_name == "LogisticRegression":
    model= LogisticRegression()
    parameters = {
        'logisticregression__n_jobs' : [-1],
        'logisticregression__solver': ['saga'],
        'logisticregression__penalty': ['l1', 'l2', None],
        'logisticregression__C': [0.1, 1, 10, 100],
    }
    task = 'classification'
    dl = False

elif model_name == "LinearRegression":
    model = LinearRegression()
    parameters = {
        'linearregression__n_jobs': [-1]
    }
    task = 'regression'
    dl = False

elif model_name == "SVC":
    model = svm.SVC()
    parameters = { 
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svc__gamma': ['scale', 'auto', 0.1, 1, 10],
        'svc__shrinking': [True, False],
    }
    task = 'classification'
    dl = False

elif model_name == "SVR":
    model = svm.SVR()
    parameters = {
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': [0.1, 1, 10],
        'svr__epsilon': [0.01, 0.1, 0.2],
        'svr__shrinking': [True, False]
    }
    task = 'regression'
    dl = False

elif model_name == "RFClassifier":
    model = RandomForestClassifier()
    parameters = {
        'randomforestclassifier__n_jobs' : [-1],
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [None, 10, 20],
        'randomforestclassifier__min_samples_split': [2, 5, 10],
        'randomforestclassifier__min_samples_leaf': [1, 2, 4],
        'randomforestclassifier__bootstrap': [True, False]
    }
    task = 'classification'
    dl = False

elif model_name == "RFRegressor":
    model = RandomForestRegressor()
    parameters = {
        'n_jobs' : [-1],
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    task = 'regression'
    dl = False

elif model_name == "ElasticNet":
    model = ElasticNet()
    parameters = {
        'alpha': [0.01, 0.1, 1.0],        # Regularization strength (higher values add more penalty)
        'l1_ratio': [0.1, 0.5, 0.9],      # Mixing parameter between L1 and L2 penalty (0: Ridge, 1: Lasso)
        'max_iter': [1000, 2000, 5000],   # Maximum number of iterations for optimization
    }
    task = 'regression'
    dl = False

elif model_name == "SGD":
    model = linear_model.SGDRegressor()
    parameters = {
        'penalty' : ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.01, 0.1, 0.3],
    }
    task = 'regression'
    dl = False

elif model_name == "deep4netClassification":
    # Create an instance of Deep4Net
    deep4net_classification = Deep4Net(
        in_chans=len(epochs.info['ch_names']),
        n_classes=n_classes_clas,
        input_window_samples=X.shape[2], 
        final_conv_length='auto',
    )
    
    model = EEGClassifier(
        module=deep4net_classification,
        criterion=torch.nn.NLLLoss,
        train_split=None, # None here, update intraining function
        callbacks = [
            "balanced_accuracy",
            "accuracy",
            ("checkpoint", Checkpoint(    dirname='best_model', f_history='best_model_checkpoint.json',
                            f_criterion=None,
                            f_optimizer=None,
                            load_best=True,
                        )),

            # ("lr_scheduler", LRScheduler(policy="ReduceLROnPlateau",
            #                              monitor="valid_loss",
            #                              patience=3)), # Another option
            ("lr_scheduler",  LRScheduler("CosineAnnealingLR", T_max=50 - 1)),
            ("early_stopping", EarlyStopping(patience=10)),

        ],
        optimizer=torch.optim.AdamW,
        optimizer__lr = 0.0001,
        optimizer__weight_decay = 0.5 * 0.001, # As recommended on braindecode.org
        batch_size = bsize,
        max_epochs=50,
        iterator_valid__shuffle=False,
        iterator_train__shuffle=True,
        device=device,
    )
    task = 'classification'
    dl = True

elif model_name == "deep4netRegression":
    # Create an instance of Deep4Net
    deep4net_regression = Deep4Net(
        in_chans=len(epochs.info['ch_names']),
        n_classes=1,
        input_window_samples=X.shape[2],
        final_conv_length='auto',
    )

    # Remove the softmax layer
    new_model = torch.nn.Sequential()
    for name, module_ in deep4net_regression.named_children():
        if "softmax" in name:
            continue
        new_model.add_module(name, module_)
    deep4net_regression = new_model

    if cuda:
        deep4net_regression.cuda()

    model = EEGRegressor(
        module=deep4net_regression,
        criterion=nn.MSELoss,
        train_split=None, # None here, update intraining function

        #cropped=True,
        #criterion=CroppedLoss,
        #criterion__loss_function=torch.nn.functional.mse_loss,
        callbacks = [
            "neg_root_mean_squared_error",
            "neg_mean_absolute_error",
            "r2",
            ("checkpoint", Checkpoint(    dirname='best_model', f_history='best_model_checkpoint.json',
                            f_criterion=None,
                            f_optimizer=None,
                            load_best=True,
                        )),

            # ("lr_scheduler", LRScheduler(policy="ReduceLROnPlateau",
            #                              monitor="valid_loss",
            #                              patience=3)), # Another option
            ("lr_scheduler",  LRScheduler("CosineAnnealingLR", T_max=50 - 1)),
            ("early_stopping", EarlyStopping(patience=10)),

        ],
        optimizer=torch.optim.AdamW,
        optimizer__lr = 0.00001,
        optimizer__weight_decay = 0, # As recommended on braindecode.org
        batch_size = bsize,
        max_epochs=50,
        iterator_valid__shuffle=False,
        iterator_train__shuffle=True,
        device=device,
    )
    task = 'regression'
    dl = True

elif model_name == "shallowFBCSPNetClassification":
    # Create an instance of ShallowFBCSPNet
    shallow_fbcsp_net_classification = ShallowFBCSPNet(
        in_chans=len(epochs.info['ch_names']),
        n_classes=n_classes_clas,
        input_window_samples=X.shape[2],
        final_conv_length='auto',
    )

    if cuda:
        shallow_fbcsp_net_classification.cuda()


    model = EEGClassifier(
        module=shallow_fbcsp_net_classification,
        criterion=torch.nn.NLLLoss,
        #train_split=None, # None here, update intraining function
        callbacks = [
            "balanced_accuracy",

            ("checkpoint", Checkpoint(    dirname='best_model', f_history='best_model_checkpoint.json',
                            f_criterion=None,
                            f_optimizer=None,
                            load_best=True,
                        )),

            # ("lr_scheduler", LRScheduler(policy="ReduceLROnPlateau",
            #                              monitor="valid_loss",
            #                              patience=3)), # Another option
            ("lr_scheduler",  LRScheduler("CosineAnnealingLR", T_max=50 - 1)),
            ("early_stopping", EarlyStopping(patience=10)),

        ],
        optimizer=torch.optim.AdamW,
        optimizer__lr = 0.00001,
        optimizer__weight_decay = 0, # As recommended on braindecode.org
        batch_size = bsize,
        max_epochs=50,
        #iterator_valid__shuffle=False,
        iterator_train__shuffle=True,
        device=device,
    )
    task = 'classification'
    dl = True


elif model_name == "shallowFBCSPNetRegression":
    # Create an instance of ShallowFBCSPNet
    shallow_fbcsp_net_regression = ShallowFBCSPNet(
        in_chans=len(epochs.info['ch_names']),
        n_classes=1,
        input_window_samples=X.shape[2],
        final_conv_length='auto',
    )

    # Remove the softmax layer
    new_model = torch.nn.Sequential()
    for name, module_ in shallow_fbcsp_net_regression.named_children():
        if "softmax" in name:
            continue
        new_model.add_module(name, module_)
    shallow_fbcsp_net_regression = new_model

    if cuda:
        shallow_fbcsp_net_regression.cuda()


    model = EEGRegressor(
        module=shallow_fbcsp_net_regression,
        criterion=nn.MSELoss,
        train_split=None, # None here, update intraining function
        #cropped=True,
        #criterion=CroppedLoss,
        #criterion__loss_function=torch.nn.functional.mse_loss,
        callbacks = [
            "neg_root_mean_squared_error",
            "r2",
            "neg_mean_absolute_error",
            ("checkpoint", Checkpoint(    dirname='best_model', f_history='best_model_checkpoint.json',
                            f_criterion=None,
                            f_optimizer=None,
                            load_best=True,
                        )),

            # ("lr_scheduler", LRScheduler(policy="ReduceLROnPlateau",
            #                              monitor="valid_loss",
            #                              patience=3)), # Another option
            ("lr_scheduler",  LRScheduler("CosineAnnealingLR", T_max=50 - 1)),
            ("early_stopping", EarlyStopping(patience=10)),

        ],
        optimizer=torch.optim.AdamW,
        optimizer__lr = 0.001,
        optimizer__weight_decay = 0, # As recommended on braindecode.org
        batch_size = bsize,
        max_epochs=50,
        iterator_valid__shuffle=False,
        iterator_train__shuffle=True,
        device=device,
    )
    task = 'regression'
    dl = True

"""elif model_name == 'covariance_MDM':
    model = make_pipeline(
                Covariances(),
                Shrinkage(),
                MDM(metric=dict(mean="riemann", distance="riemann")),
            )
    parameters = {
        'shrinkage__shrinkage': [0.1, 0.2, 0.5, 0.8],
        'mdm__metric': [
            {'mean': 'riemann', 'distance': 'riemann'},
            {'mean': 'riemann', 'distance': 'logeuclid'},
            {'mean': 'logeuclid', 'distance': 'riemann'},
            {'mean': 'logeuclid', 'distance': 'logeuclid'},
        ],
        'mdm__n_jobs': [-1],
    }
    task = 'classification'
    dl = False"""

print(model_name, part)
if dl == True:
    search_params = False

# Set y
if task == 'classification':
    target == '3_classes' #take this out later! Just for now, to avoid mix up
    epochs.metadata['task'].astype(str)
    if target == '3_classes':
        y = [i.replace('rate', '') for i in epochs.metadata["task"].values]
        y = np.array(y)
    elif target == '5_classes':
        y = epochs.metadata["task"].values
elif task == 'regression':
    target == 'intensity' #take this out later! Just for now, to avoid mix up
    if target == 'rating':
        y = epochs.metadata["rating"].values 
    elif target == 'intensity':
        y = epochs.metadata["intensity"].values 

#_______________________________________________________________________________
#Training here

#from skorch.utils import load_checkpoint

best_params = {
        'penalty' : 'l2',
        'alpha': 0.01,
    }


if dl:
    # Convert categorical labels to integer indices
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        y = torch.tensor(y, dtype=torch.int64)
        # Print the mapping between integer labels and original labels
        label_mapping = dict(zip(np.unique(y), label_encoder.classes_))
        print("Label mapping:")
        for integer_label, original_label in label_mapping.items():
            print(f"{integer_label} -> {original_label}")        
    else:
        # Convert numerical labels to float
        y = torch.tensor(y, dtype=torch.float32)

    X = X.astype(np.int64)

    model.fit(X, y)

    # Save the final model to a file
    joblib.dump(model, f'{model_dir}{model_name}_{target}.joblib')
    #torch.save(model.state_dict(), f'{model_dir}_{model_name}_{target}.pth')



elif search_params:
    if task == "regression":
        scoring = 'neg_root_mean_squared_error'
    else:
        scoring = 'accuracy'

    if isinstance(model, Pipeline):
        pass
    else:
        vectorizer = mne.decoding.Vectorizer()
        X = vectorizer.fit_transform(X)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(n_jobs=-1, estimator=model, param_grid=parameters, cv=10, scoring=scoring)

    # Fit (train) the GridSearchCV object on your data
    grid_search.fit(X, y)

    # Get the best parameters and the best estimator (model)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Optionally, you can save the best model to a file
    joblib.dump(best_model, f'{model_dir}{model_name}_{target}.joblib')

else:
    # Initialize your machine learning model with specific parameters
    best_model = model.set_params(**best_params)

    if isinstance(model, Pipeline):
        pass
    else:
        vectorizer = mne.decoding.Vectorizer()
        X = vectorizer.fit_transform(X)

    # Fit (train) the model on the full dataset
    best_model.fit(X, y)

    # Save the model to a file
    joblib.dump(best_model, f'{model_dir}{model_name}_{target}.joblib')
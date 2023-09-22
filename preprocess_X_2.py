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

from sklearn.pipeline import make_pipeline
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
import os
from sklearn.linear_model import ElasticNet
import sys
from braindecode.preprocessing import exponential_moving_standardize
#from pyriemann.classification import MDM, TSclassifier
#from pyriemann.estimation import Covariances, Shrinkage

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
    if torch.cuda.is_available():
        device = torch.device('cuda')  # PyTorch will use the default GPU
        torch.backends.cudnn.benchmark = True
    else:
        cuda = False
        device = torch.device('cpu')

    bidsroot = '/lustre04/scratch/mabus103/2023_eegmarkers/derivatives/epochs_clean/cleaned_epo.fif'
    log_dir=f'/lustre04/scratch/mabus103/logs2'
    #log_dir=f'/lustre04/scratch/mabus103/ML_for_Pain_Prediction/logs'
elif 'media/mp' in current_directory: #MP's local machine
    model_name = "shallowFBCSPNetClassification"
    part = "between"
    target = "3_classes"
    bsize = 16
    device = torch.device('cuda')
    bidsroot = 'data/cleaned_epo.fif'
    log_dir= 'logs'

elif "mplab" in current_directory:
    model_name = "SGD" #set the model to use. also determines dl and kind of task
    part = 'between' # 'between' or 'within' participant
    target = "3_classes"
    optimizer_lr = 0.000625
    bsize = 16
    device = torch.device('cpu')  # Use CPU if GPU is not available or cuda is False
    #bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/cleaned_epo.fif'
    bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
    log_dir='/home/mplab/Desktop/Mathilda/Project/code/ML_for_Pain_Prediction/logs'
else:
    model_name = "shallowFBCSPNetRegression" #set the model to use. also determines dl and kind of task
    part = 'between'# 'between' or 'within' participant
    target = "intensity"
    optimizer_lr = 0.000625
    bsize = 16
    device = torch.device('cpu')  # Use CPU if GPU is not available or cuda is False
    bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
    log_dir='/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs2'

data_path = opj(bidsroot)
# Load epochs oject
epochs = mne.read_epochs(data_path, preload=True)


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

X = epochs.get_data()
X = X*1e6 # Convert from V to uV

for epo in tqdm(range(X.shape[0]), desc='Normalizing data'): # Loop epochs
    X[epo, :, :] = exponential_moving_standardize(X[epo, :, :], factor_new=0.001, init_block_size=None) # Normalize the data

# Save the preprocessed data and additional information to a .npz file
np.savez('normalized_X_2.npz', X=X)

# Save the selected indices to a text file
with open("selected_indices2.txt", "w") as f:
    for index in selected_indices:
        f.write(f"{index}\n")

epochs.save(opj('normalized_epo2.fif'), overwrite=True)
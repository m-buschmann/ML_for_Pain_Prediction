import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import mne
from os.path import join as opj
import torch
from sklearn import datasets, linear_model, svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from braindecode.models import ShallowFBCSPNet,Deep4Net, EEGNetv4
from braindecode import EEGClassifier, EEGRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler, ProgressBar, EpochScoring
import torch.nn as nn
import pandas as pd
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
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score



# Get the current working directory
current_directory = os.getcwd()

# Check if the keyword "lustre" is present in the current directory
cuda = "lustre" in current_directory #this is changed to false if no GPU
cc = "lustre" in current_directory

# If cuda is True and a GPU is available, set up GPU acceleration in PyTorch
# And set bidsroot according to device 
if cuda:
    model_name = sys.argv[1]
    bsize = 16  
    target = sys.argv[2]
    if torch.cuda.is_available():
        device = torch.device('cuda')  # PyTorch will use the default GPU
        torch.backends.cudnn.benchmark = True
    else:
        cuda = False
        device = torch.device('cpu')

    bidsroot = '/lustre04/scratch/mabus103/normalized_data/normalized_epo.fif'
    log_dir=f'/lustre04/scratch/mabus103/logs'
    model_dir=f'/lustre04/scratch/mabus103/models'
elif 'media/mp' in current_directory: #MP's local machine
    model_name = "shallowFBCSPNetClassification"
    target = "3_classes"
    bsize = 16
    device = torch.device('cuda')
    bidsroot = 'data/cleaned_epo.fif'
    log_dir= 'logs'

elif "mplab" in current_directory:
    model_name = "SGD" #set the model to use. also determines dl and kind of task
    target = "intensity"
    bsize = 16
    device = torch.device('cpu')  # Use CPU if GPU is not available or cuda is False
    #bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/cleaned_epo.fif'
    bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
    log_dir='/home/mplab/Desktop/Mathilda/Project/code/ML_for_Pain_Prediction/logs'
    model_dir='/home/mplab/Desktop/Mathilda/Project/code/ML_for_Pain_Prediction/models'
else:
    model_name = "shallowFBCSPNetClassification" #set the model to use. also determines dl and kind of task
    target = "intensity"
    task = "regression"
    search_params = True
    dl = False
    bsize = 16
    device = torch.device('cpu')  # Use CPU if GPU is not available or cuda is False
    bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
    log_dir='/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs'
    model_dir='/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/models'

data_path = opj(bidsroot)

# Load epochs oject
epochs = mne.read_epochs(data_path, preload=True)

# Load the saved model
model = joblib.load(model_dir)

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
else:
    if isinstance(model, Pipeline):
        pass
    else:
        vectorizer = mne.decoding.Vectorizer()
        X = vectorizer.fit_transform(X)

X_test= X[0:100] #just for testing code on my laptop from 0 to 100
y_test = y[0:100]    

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate score
if task == "classification":
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy on Test Data: {accuracy:.2f}")
else:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE on Test Data: {rmse:.2f}")
    r2 = r2_score(y_test, y_pred)


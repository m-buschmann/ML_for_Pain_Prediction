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
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


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

    bidsroot = '/lustre04/scratch/mabus103/2normalized_data/2normalized_epo.fif'
    log_dir=f'/lustre04/scratch/mabus103/2logs'
    model_dir=f'/lustre04/scratch/mabus103/trained_models'
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
    target = "3_classes"
    search_params = True
    bsize = 16
    device = torch.device('cpu')  # Use CPU if GPU is not available or cuda is False
    bidsroot = '/home/mathilda/MITACS/Project/2023_eegmarkers/derivatives/epochs_clean/cleaned_epo_sub2_to_4.fif'
    log_dir='/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/2logs'
    model_dir='/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/trained_models'

data_path = opj(bidsroot)
# Load epochs oject, is already normalized and some epochs removed on compute canada
epochs = mne.read_epochs(data_path, preload=True)


if model_name == "deep4netClassification":
    model = "deep4netClassification_3_classes.joblib"
    dl = True
    task = "classification"
elif model_name == "deep4netRegression":
    model = "deep4netRegression_intensity.joblib"
    dl = True
    task = "regression"
elif model_name == "shallowFBCSPNetClassification":
    #model = "shallowFBCSPNetClassification_3_classes.joblib"
    model = "shallowFBCSPNetClassification_3_classes.pth"
    dl = True
    task = "classification"
elif model_name == "shallowFBCSPNetRegression":
    model = "shallowFBCSPNetRegression_intensity.joblib"
    dl = True
    task = "regression"
elif model_name == "RFClassifier":
    model = "RFClassifier_3_classes.joblib"
    dl = False
    task = "classification"
elif model_name == "RFRegressor":
    #... model = ...
    dl = False
    task = "regression"

model_path = opj(model_dir, model)

if cc:
    # load already normalized X
    loaded_data = np.load('/lustre04/scratch/mabus103/2normalized_data/2normalized_X.npz')  # For .npz format
    X = loaded_data['X']

    # Only use thermal for regression
    if target == "intensity" or target == "rating":
        # Load the metadata information from the original epochs object
        metadata_df = epochs.metadata

        # Find the indices where the task is "thermal" or "thermalrate"
        thermal_indices = np.where(metadata_df["task"].isin(["thermalactive", "thermalpassive"]))[0]

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

    epochs = epochs.pick_types(eeg=True)
    num_channels = len(epochs.info['ch_names'])
    print(f"Number of channels in the Epochs object: {num_channels}")

    if target == "intensity" or target == "rating":
        #only use thermal task for pain intensity
        selected_tasks = ["thermalactive", "thermalpassive"]
        epochs = epochs[epochs.metadata["task"].isin(selected_tasks)]
        X = epochs.get_data()
    else:
        X = epochs.get_data()

    X = X*1e6 # Convert from V to uV  
    for epo in tqdm(range(X.shape[0]), desc='Normalizing data'): # Loop epochs
        X[epo, :, :] = exponential_moving_standardize(X[epo, :, :], factor_new=0.001, init_block_size=None) # Normalize the data

# Define the groups (participants) to avoid splitting them across train and test
groups = epochs.metadata["participant_id"].values
print (model_name)

# Set y
if task == "classification":
    epochs.metadata['task'].astype(str)
    if target == '3_classes':
        y = [i.replace('active', '') for i in epochs.metadata["task"].values]
        y = [i.replace('passive', '') for i in epochs.metadata["task"].values]
        y = np.array(y)
    elif target == '5_classes':
        y = epochs.metadata["task"].values
    elif target == 'pain':
        y_values = []
        for index, row in epochs.metadata.iterrows():
                if row['intensity'] >= 100 and (row['task'] == 'thermalactive' or row['task'] == 'thermalpassive'):
                    y_values.append("pain")
                else:
                    y_values.append("no pain")
        y = np.array(y_values)
    elif target == 'pain_with_us':
        y_values = []
        for index, row in epochs.metadata.iterrows():
                if row['intensity'] >= 100 and (row['task'] == 'thermalactive' or row['task'] == 'thermalpassive'):
                    y_values.append("pain")
                else:
                    y_values.append("no pain")
        y = np.array(y_values)
        # Only use as much 'no pain' data as 'pain' data
        print('Original dataset shape %s' % Counter(y))
        X_flat = X.reshape(X.shape[0], -1)
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y = rus.fit_resample(X_flat, y)
        n = len(y)
        X = X_resampled.reshape(n, X.shape[1], X.shape[2])
        print('Resampled dataset shape %s' % Counter(y))
        # Get indices that are kept in the data
        selected_indices = rus.sample_indices_
        # Use these indices to filter the 'groups' array
        groups = groups[selected_indices]

elif task == 'regression':
    if target == 'rating':
        y = epochs.metadata["rating"].values 
    elif target == 'intensity':
        y = epochs.metadata["intensity"].values 

# Check for same length
print("groups:", len(groups))
print("X:",len(X))
print("y:",len(y))

if dl:

    if cc:
        model = torch.load(model_path, map_location=torch.device('cuda'))
    else:
        # Load the saved model
        model = torch.load(model_path, map_location=torch.device('cpu'))

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

# Make predictions on the test data
y_pred = model.predict(X)

# Calculate score
if task == "classification":
    accuracy = accuracy_score(y, y_pred)

    print(f"Model Accuracy on Test Data: {accuracy:.2f}")
else:
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"RMSE on Test Data: {rmse:.2f}")
    r2 = r2_score(y, y_pred)
    print(f"r2 on Test Data: {r2:.2f}")


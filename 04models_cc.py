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
from bayes_train_script_between_part import bayes_training_nested_cv_between
from bayes_train_script_within_part import bayes_training_nested_cv_within

import torch.nn as nn
import pandas as pd
from sklearn.pipeline import make_pipeline
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
import os
import json
from sklearn.linear_model import ElasticNet
import sys
from braindecode.preprocessing import exponential_moving_standardize
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
#from pyriemann.classification import MDM, TSclassifier
#from pyriemann.estimation import Covariances, Shrinkage

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
    if torch.cuda.is_available():
        device = torch.device('cuda')  # PyTorch will use the default GPU
        torch.backends.cudnn.benchmark = True
    else:
        cuda = False
        device = torch.device('cpu')

    bidsroot = '/lustre04/scratch/mabus103/normalized_data/normalized_epo.fif'
    log_dir=f'/lustre04/scratch/mabus103/logs'
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
    model_name = "deep4netClassification" #set the model to use. also determines dl and kind of task
    part = 'within'# 'between' or 'within' participant
    target = "pain" #intensity, rating, 3_classes, 5_classes, pain, pain_with_us
    optimizer_lr = 0.000625
    bsize = 16
    device = torch.device('cpu')  # Use CPU if GPU is not available or cuda is False
    bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
    log_dir='/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs'

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

    epochs = epochs.pick_types(eeg=True)
    num_channels = len(epochs.info['ch_names'])
    print(f"Number of channels in the Epochs object: {num_channels}")

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
elif target == "pain" or target == "pain_with_us":
    n_classes_clas = 2


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
        'randomforestregressor__n_jobs' : [-1],
        'randomforestregressor__n_estimators': [50, 100, 200],
        'randomforestregressor__max_depth': [None, 10, 20],
        'randomforestregressor__min_samples_split': [2, 5, 10],
        'randomforestregressor__min_samples_leaf': [1, 2, 4],
        'randomforestregressor__bootstrap': [True, False]
    }
    task = 'regression'
    dl = False

elif model_name == "ElasticNet":
    model = ElasticNet()
    parameters = {
        'elasticnet__alpha': [0.01, 0.1, 1.0],        # Regularization strength (higher values add more penalty)
        'elasticnet__l1_ratio': [0.1, 0.5, 0.9],      # Mixing parameter between L1 and L2 penalty (0: Ridge, 1: Lasso)
        'elasticnet__max_iter': [1000, 2000, 5000],   # Maximum number of iterations for optimization
    }
    task = 'regression'
    dl = False

elif model_name == "SGD":
    model = linear_model.SGDRegressor()
    parameters = {
        'sgdregressor__penalty' : ['l2', 'l1', 'elasticnet'],
        'sgdregressor__alpha': [0.0001, 0.01, 0.1, 0.3],
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
        train_split=None, # None here, update in training function
        callbacks = [
            "balanced_accuracy",
            "accuracy",
            ("checkpoint", Checkpoint(
                            f_criterion=None,
                            f_optimizer=None,
                            f_history=None,
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
            ("checkpoint", Checkpoint(
                            f_criterion=None,
                            f_optimizer=None,
                            f_history=None,
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
        train_split=None, # None here, update intraining function
        callbacks = [
            "balanced_accuracy",

            ("checkpoint", Checkpoint(
                            f_criterion=None,
                            f_optimizer=None,
                            f_history=None,
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
            ("checkpoint", Checkpoint(
                            f_criterion=None,
                            f_optimizer=None,
                            f_history=None,
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

#_____________________________________________________________________-
# Training

# Set y
if task == 'classification':
    epochs.metadata['task'].astype(str)
    if target == '3_classes':
        y = [i.replace('rate', '') for i in epochs.metadata["task"].values]
        y = np.array(y)
    elif target == '5_classes':
        y = epochs.metadata["task"].values
    elif target == 'pain':
        y_values = []
        for index, row in epochs.metadata.iterrows():
                if row['intensity'] >= 100 and (row['task'] == 'thermal' or row['task'] == 'thermalrate'):
                    y_values.append("pain")
                else:
                    y_values.append("no pain")
        y = np.array(y_values)
    elif target == 'pain_with_us':
        y_values = []
        for index, row in epochs.metadata.iterrows():
                if row['intensity'] >= 100 and (row['task'] == 'thermal' or row['task'] == 'thermalrate'):
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

# Get writer for tensorboard
writer = SummaryWriter(log_dir=opj(log_dir, model_name, f"{part}_{target}"))

# Train the EEG model using cross-validation
if dl == False and part == 'within':
    mean_score, all_true_labels, all_predictions, participant_scores, most_common_best_param = training_nested_cv_within(model, X, y, parameters, task=task, nfolds=10, n_inner_splits=5, groups=groups, writer=writer)
if dl == False and part == 'between':
    mean_score, all_true_labels, all_predictions, score_test, most_common_best_param = training_nested_cv_between(model, X, y, parameters = parameters, task =task, nfolds=10, n_inner_splits=5, groups=groups, writer=writer)
if dl == True and part == 'within':
    mean_score, all_true_labels, all_predictions, participants_scores = trainingDL_within(model, X, y, task=task, groups=groups, writer=writer, nfolds=10)
if dl == True and part == 'between':
    mean_score, all_true_labels, all_predictions, score_test = trainingDL_between(model, X, y, task=task, nfolds=10, groups=groups, writer=writer)

# Close the SummaryWriter when done
writer.close()

# Specify the file path for storing the results
output_dir = f"results{model_name}"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_file = os.path.join(output_dir, f"{part}_{target}.csv")

# Determine the length of the data
data_length = len(all_true_labels)
if dl == True and part == "within":
    # Create a DataFrame
    data = pd.DataFrame({
        "Mean Score": [mean_score] + ["_"] * (data_length - 1),
        "Participant Scores": participants_scores + ["_"] * (data_length - len(participants_scores)),
        "True Label": all_true_labels,
        "Predicted Label": all_predictions
    })

    # Write the DataFrame to a CSV file
    data.to_csv(output_file, index=False)

elif dl == True and part == "between":
    # Create a DataFrame
    data = pd.DataFrame({
        "Mean Score": [mean_score] + ["_"] * (data_length - 1),
        "Test Scores": [score_test] + ["_"] * (data_length - 1),
        "True Label": all_true_labels,
        "Predicted Label": all_predictions
    })

    # Write the DataFrame to a CSV file
    data.to_csv(output_file, index=False)


elif dl == False and part == "between":
    # Convert the dictionary to a JSON string and remove commas
    most_common_best_param_json = json.dumps(most_common_best_param, separators=(',', ':'))
    # Replace commas with semicolons
    most_common_best_param_json = most_common_best_param_json.replace(',', ';')
    
    # Create a DataFrame
    data = pd.DataFrame({
        "Mean Score": [mean_score] + ["_"] * (data_length - 1),
        "Test Scores": score_test + ["_"] * (data_length - len(score_test)),
        "Most common best Parameter": [most_common_best_param_json] + ["_"] * (data_length - 1),
        "True Label": all_true_labels,
        "Predicted Label": all_predictions
    })

    # Write the DataFrame to a CSV file
    data.to_csv(output_file, index=False)

elif dl == False and part == "within":
    # Convert the dictionary to a JSON string and remove commas
    most_common_best_param_json = json.dumps(most_common_best_param, separators=(',', ':'))
    # Replace commas with semicolons
    most_common_best_param_json = most_common_best_param_json.replace(',', ';')
    
    # Create a DataFrame
    data = pd.DataFrame({
        "Mean Score": [mean_score] + ["_"] * (data_length - 1),
        "Participant Mean Scores": participant_scores + ["_"] * (data_length - len(participant_scores)),
        "Most common best Parameter": [most_common_best_param_json] + ["_"] * (data_length - 1),
        "True Label": all_true_labels,
        "Predicted Label": all_predictions
    })

    # Write the DataFrame to a CSV file
    data.to_csv(output_file, index=False)


# For classification, build a confusion matrix
if task == 'classification':
    # Load data from the CSV file
    data = pd.read_csv(output_file)  
    true_labels = data['True Label']
    predicted_labels = data['Predicted Label']

    # Get unique class labels
    classes = sorted(list(set(true_labels) | set(predicted_labels)))

    # Create a confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)

    print("Confusion Matrix:")
    print(cm)

    # Plot and save the confusion matrix
    # Compute the confusion matrix
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = "{:.2f}".format(cm_normalized[i, j])
            plt.text(j, i, text, ha='center', va='center', color='white')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout(pad=1.5)

    # Add a legend with custom labels
    if dl == True and target == "5_classes":
        target_names = ["auditory", "auditoryrate", "rest", "thermal", "thermalrate"]
        legend_labels = {
            0: 'auditory',
            1: "auditoryrate",
            2: 'rest',
            3: 'thermal',
            4: "thermalrate"
        }

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f"{label}: {text}", markersize=10, markerfacecolor='b') for label, text in legend_labels.items()]
        plt.legend(handles=legend_elements, title="Legend")
    
    elif dl == True and target  == "3_classes":
        legend_labels = {
            0: 'auditory',
            1: 'rest',
            2: 'thermal'
        }

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f"{label}: {text}", markersize=10, markerfacecolor='b') for label, text in legend_labels.items()]
        plt.legend(handles=legend_elements, title="Legend")
    
    elif dl == True and (target == 'pain' or target == "pain_with_us"):
        target_names = ["no pain", "pain"]
        legend_labels = {
            0: 'no pain',
            1: "pain",
        }

    plt.show()

    # Save the confusion matrix plot as an image file
    output_dir = f"images/confusion_matrix{model_name}"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    output_file = os.path.join(output_dir, f"{part}_{target}.png")
    plt.savefig(output_file)

# Run this in Terminal to see tensorboard
#tensorboard --logdir /home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs/deep4netClassification/between --port 6007
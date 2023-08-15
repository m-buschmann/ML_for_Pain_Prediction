import mne
from os.path import join as opj
import torch
from sklearn import datasets, linear_model, svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GroupKFold, KFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from collections import Counter
from braindecode.models import ShallowFBCSPNet,Deep4Net, EEGNetv4
from braindecode import EEGClassifier, EEGRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler, ProgressBar, EpochScoring
from train_script_between_part import trainingDL_between, training_nested_cv_between
from train_script_within_part import training_nested_cv_within, trainingDL_within
import torch.nn as nn
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.training.losses import CroppedLoss
from torch.nn import MSELoss
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# Set kind of Cross validation and task to perform 
part = 'between' # 'between' or 'within' participant
cv = 'simple' # 'simple' or 'nested' Cross Validation
task = 'classification' # 'classification' or 'regression'
dl = True # Whether to use a deep learning or standard ML model

#____________________________________________________________________________
# Application of cross validation for different models
# Load data

# Directory
#bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/cleaned_epo.fif'
#bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
data_path = opj(bidsroot)
# Load epochs oject
epochs = mne.read_epochs(data_path, preload=True)
# Exclude eog and misc channels
epochs = epochs.pick_types(eeg=True) 

# Set target and label data
X = epochs.get_data()

# Rescale X to a bigger number
X = X * 10e6

if task == 'classification':
    y = epochs.metadata["task"].values  
elif task == 'regression':
    y = epochs.metadata["rating"].values #maybe also intensity


# Define the groups (participants) to avoid splitting them across train and test
groups = epochs.metadata["participant_id"].values


#____________________________________________________________________
# Create models: Classification

n_chans = len(epochs.info['ch_names'])
input_window_samples=X.shape[2]
n_classes_clas=5
bsize = 4


# Define a balanced accuracy
def balanced_accuracy(model, X, y=None):
    # assume ds yields (X, y), e.g. torchvision.datasets.MNIST
    y_true = [y for _, y in X]
    y_pred = model.predict(X)
    return balanced_accuracy_score(y_true, y_pred)

# Create an instance of ShallowFBCSPNet
shallow_fbcsp_net = ShallowFBCSPNet(
    in_chans=len(epochs.info['ch_names']),
    n_classes=n_classes_clas,
    input_window_samples=X.shape[2],
    final_conv_length='auto',
)
model_name = "shallowFBCSPNetClassification"

# Create an instance of Deep4Net
deep4net = Deep4Net(
    in_chans=len(epochs.info['ch_names']),
    n_classes=n_classes_clas,
    input_window_samples=X.shape[2],
    final_conv_length='auto',
)
#model_name = "deep4netClassification"

# Create EEGClassifiers

model = EEGClassifier(
    module=shallow_fbcsp_net,
    callbacks = [
        Checkpoint,
        EarlyStopping,
        LRScheduler,
        #ProgressBar,
        EpochScoring(scoring=balanced_accuracy, lower_is_better=False),
    ],
    optimizer=torch.optim.Adam,
    batch_size = bsize,
    max_epochs=20,
)

#model= LogisticRegression()
#model_name = "LogisticRegression"

#model = svm.SVC()
#model_name = "SVC"

#model = RandomForestClassifier()
#model_name = "RFClassifier"

#____________________________________________________________________
# Create EEGRegressors

#batchsize 8 -16
optimizer_lr = 0.000625
optimizer_weight_decay = 0
n_classes_reg=1

# Create an instance of ShallowFBCSPNet
shallow_fbcsp_net = ShallowFBCSPNet(
    in_chans=len(epochs.info['ch_names']),
    n_classes=n_classes_reg,
    input_window_samples=X.shape[2],
    final_conv_length='auto',
)
#model_name = "shallowFBCSPNetRegression"

# Create an instance of Deep4Net
deep4net = Deep4Net(
    in_chans=len(epochs.info['ch_names']),
    n_classes=n_classes_reg,
    input_window_samples=X.shape[2],
    final_conv_length='auto',
)

"""model = EEGRegressor(
    module=deep4net,
    criterion=MSELoss(),
    #cropped=True,
    #criterion=CroppedLoss,
    #criterion__loss_function=torch.nn.functional.mse_loss,
    callbacks = [
        'neg_root_mean_squared_error',
        Checkpoint(load_best=True),
        EarlyStopping,
        LRScheduler,
        ProgressBar,
    ],
    optimizer=torch.optim.Adam,
    batch_size = bsize,
    max_epochs=20,
)
model_name = "deep4netRegression"
"""
#model = svm.SVR()
#model_name = "SVR"

#model = RandomForestRegressor()
#model_name = "RFRegressor"

#model = LinearRegression()  
#model_name = "LinearRegression"

#__________________________________________________________________
# Training

# Choose parameters for nested CV
if model_name == "LinearRegression":
    parameters = {
        'fit_intercept': [True, False],
        'positive': [True, False]
    }
elif model_name == "LogisticRegression":
    parameters = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.1, 1, 10, 100],
        'multi_class': ['ovr', 'multinomial'],
        'class_weight': [None, 'balanced']
    }
elif model_name == "SVC":
    parameters = { 
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.1, 1, 10],
        'shrinking': [True, False],
        'tol': [1e-3, 1e-4, 1e-5],
        'class_weight': [None, 'balanced'],
        'decision_function_shape': ['ovr', 'ovo'],
    }
elif model_name == "SVR":
    parameters = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2],
        'shrinking': [True, False]
    }
elif model_name == "RFClassifier":
    parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False]
    }
elif model_name == "RFRegressor":
    parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False]
    }


# Get writer for tensorboard
#writer = SummaryWriter(log_dir=f'/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs/{model_name}/{part}')
writer= SummaryWriter(log_dir=f'/home/mplab/Desktop/Mathilda/Project/code/ML_for_Pain_Prediction/logs/{model_name}/{part}')

# Train the EEG model using cross-validation
if dl == False and part == 'within':
    mean_score, all_true_labels, all_predictions, score_test, most_common_best_param = training_nested_cv_within(model, X, y, parameters, task=task, groups=groups, writer=writer)
if dl == False and part == 'between':
    mean_score, all_true_labels, all_predictions, score_test, most_common_best_param = training_nested_cv_between(model, X, y, parameters = parameters, task =task, nfolds=3, groups=groups, writer=writer)
if dl == True and part == 'within':
    mean_score, all_true_labels, all_predictions, score_test = trainingDL_within(model, X, y, task=task, groups=groups, writer=writer)
if dl == True and part == 'between':
    mean_score, all_true_labels, all_predictions, score_test = trainingDL_between(model, X, y, task=task, nfolds=3, groups=groups, writer=writer)

# Close the SummaryWriter when done
writer.close()

# Specify the file path for storing the results
output_dir = f"results{model_name}"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_file = os.path.join(output_dir, f"{part}.csv")

# Combine the lists into rows
if cv == 'simple':
    rows = zip([mean_score], [score_test], all_true_labels, all_predictions)

    # Write the rows to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Mean Score", "Test Score", "True Label", "Predicted Label"])  # Write header
        for row in rows:
            mean_scr, test_scr, true_labels, preds = row
            csvwriter.writerow([mean_scr, test_scr, true_labels.tolist(), preds.tolist()])

elif cv == 'nested':
    rows = zip([mean_score], [score_test], [most_common_best_param], all_true_labels, all_predictions)

    # Write the rows to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Mean Score", "Test Score", "Best Parameters", "True Label", "Predicted Label"])  # Write header
        csvwriter.writerows(rows)

# For classification, build a confusion matrix
if task == 'classification':
    target_names = ["thermalrate", "auditoryrate", "thermal", "auditory", "rest"]

    # Convert the lists to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)

    # Compute the confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    fig, ax = plt.subplots(1)
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(title="Normalized Confusion matrix")
    fig.colorbar(im)
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    fig.tight_layout()
    ax.set(ylabel="True label", xlabel="Predicted label")

    # Save the confusion matrix plot as an image file
    output_dir = f"images/confusion_matrix{model_name}"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    output_file = os.path.join(output_dir, f"{part}.png")
    plt.savefig(output_file)

# Run this in Terminal to see tensorboard
#tensorboard --logdir /home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs/deep4netRegression/between --port 6007

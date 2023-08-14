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


#____________________________________________________________________________
# Application of cross validation for different models
# Load data

# Directory
#bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/cleaned_epo.fif'
bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
data_path = opj(bidsroot)
# Load epochs oject
epochs = mne.read_epochs(data_path, preload=True)
# Exclude eog and misc channels
epochs = epochs.pick_types(eeg=True) 

# Set target and label data
X = epochs.get_data()

# Rescale X to a bigger number
X = X * 10e6
y = epochs.metadata["rating"].values #maybe also intensity
#y = epochs.metadata["task"].values

# Define the groups (participants) to avoid splitting them across train and test
groups = epochs.metadata["participant_id"].values


#____________________________________________________________________
# Create models: Classification

n_chans = len(epochs.info['ch_names'])
n_classes_clas=5
input_window_samples=X.shape[2]

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
#model_name = "shallowFBCSPNetClassification"

# Create an instance of Deep4Net
deep4net = Deep4Net(
    in_chans=len(epochs.info['ch_names']),
    n_classes=n_classes_clas,
    input_window_samples=X.shape[2],
    final_conv_length='auto',
)
#model_name = "deep4netClassification"

# Create EEGClassifiers
"""
model = EEGClassifier(
    module=shallow_fbcsp_net,
    callbacks = [
        Checkpoint,
        EarlyStopping,
        LRScheduler,
        ProgressBar,
        EpochScoring(scoring=balanced_accuracy, lower_is_better=False),
    ],
    optimizer=torch.optim.Adam,
    max_epochs=20,
)"""

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

model = EEGRegressor(
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
    max_epochs=20,
)
model_name = "deep4netRegression"

#model = svm.SVR()
#model_name = "SVR"

#model = RandomForestRegressor()
#model_name = "RFRegressor"

#model = LinearRegression()  
#model_name = "LinearRegression"

#__________________________________________________________________
# Training

# Choose parameters for nested CV
if model_name== "LinearRegression":
    parameters = {"C": [1, 10, 100]}

#parameters = {"C": [1, 10, 100]}
#parameters = {"n_estimators": [1, 10, 100]}
#parameters = {"fit_intercept": [True, False]}


# Train the EEG model using cross-validation
# Get writer for tensorboard
writer = SummaryWriter(log_dir=f'/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs/{model_name}/{cv}')
cv = 'between'

#mean_score, all_true_labels, all_predictions, score_test, most_common_best_param = training_nested_cv_within(model, X, y, parameters, task='regression', groups=groups)
#mean_score, all_true_labels, all_predictions, score_test, most_common_best_param = training_nested_cv_between(model, X, y, parameters = parameters, task = 'regression', nfolds=3, groups=groups)

#mean_score, all_true_labels, all_predictions = trainingDL_within(model, X, y, task='regression', groups=groups)
mean_score, all_true_labels, all_predictions, score_test = trainingDL_between(model, X, y, task='regression', nfolds=3, groups=groups, writer=writer)

# Close the SummaryWriter when done
writer.close()

# Run this in Terminal
#tensorboard --logdi/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs/deep4net
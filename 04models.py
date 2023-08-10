import mne
from os.path import join as opj
import torch
from sklearn import datasets, linear_model, svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GroupKFold, KFold, GridSearchCV
from collections import Counter
from braindecode.models import ShallowFBCSPNet, EEGNetv4
from braindecode import EEGClassifier
from sklearn.ensemble import RandomForestClassifier
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler, ProgressBar
from train_script_between_part import trainingDL_between, training_nested_cv_between
from train_script_within_part import training_nested_cv_within, trainingDL_within

#TODO: compare first models
#implement more
#put all y on lists

#____________________________________________________________________________
#application of cross validation for different models

# Directory
bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
data_path = opj(bidsroot)
# Load epochs oject
epochs = mne.read_epochs(data_path, preload=True)

# Set target and label data
X = epochs.get_data()
# Rescale X to a bigger number
X = X * 10e6
#y = epochs.metadata["rating"].values # .value correct? else, missing epochs are a problem
y = epochs.metadata["task"].values

# Define the groups (participants) to avoid splitting them across train and test
groups = epochs.metadata["participant_id"].values

# Create an instance of ShallowFBCSPNet
shallow_fbcsp_net = ShallowFBCSPNet(
    in_chans=len(epochs.info['ch_names']),
    n_classes=5,
    #filter_time_length = 1,
    input_window_samples=1000,
    final_conv_length='auto',
)

# Create EEGClassifier with ShallowFBCSPNet as the model
model = EEGClassifier(
    module=shallow_fbcsp_net,
    callbacks = [
        #balanced acc
        Checkpoint,
        EarlyStopping,
        LRScheduler,
        ProgressBar,
    ],
    optimizer=torch.optim.Adam,
    max_epochs=20,
)
parameters = {"C": [1, 10, 100]}

"""model_params = {
    'in_chans': len(epochs.info['ch_names']),
    'n_classes': 5,
    #'filter_time_length' : 1, #non negotiable
    'input_window_samples': 1000,
    'pool_mode' : 'mean',
    'final_conv_length': 'auto',
}

model = NeuralNetClassifier(
    EEGNetv4,
    module__in_chans=model_params['in_chans'],
    module__n_classes=model_params['n_classes'],
    #module__filter_time_length=model_params['filter_time_length'],
    module__input_window_samples=model_params['input_window_samples'],
    module__pool_mode=model_params['pool_mode'],
    module__final_conv_length=model_params['final_conv_length'],
    optimizer=torch.optim.Adam,  # Set the optimizer here
    max_epochs=20,  # Set the number of epochs
)"""

#model= LogisticRegression()   
#model = svm.SVC()
parameters = {"C": [1, 10, 100]}

#model = RandomForestClassifier()
#parameters = {"n_estimators": [1, 10, 100]}

mean_score, all_true_labels, all_predictions, score_test = trainingDL_within(model, X, y, task='classification', groups=groups)
#mean_score, all_true_labels, all_predictions, score_test, most_common_best_param = training_nested_cv(model, X, y, parameters = parameters, task = 'classification', nfolds=3, groups=groups)

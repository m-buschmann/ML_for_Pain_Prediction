import mne
from os.path import join as opj
import torch
from sklearn import datasets, linear_model, svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GroupKFold, KFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from collections import Counter
from braindecode.models import ShallowFBCSPNet, EEGNetv4
from braindecode import EEGClassifier
from sklearn.ensemble import RandomForestClassifier
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler, ProgressBar, EpochScoring
from train_script_between_part import trainingDL_between, training_nested_cv_between
from train_script_within_part import training_nested_cv_within, trainingDL_within

#TODO: compare first models
#implement more

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
    input_window_samples=X.shape[2],
    final_conv_length='auto',
)

# Define a balanced accuracy
def balanced_accuracy(model, X, y=None):
    # assume ds yields (X, y), e.g. torchvision.datasets.MNIST
    y_true = [y for _, y in X]
    y_pred = model.predict(X)
    return balanced_accuracy_score(y_true, y_pred)

# Create EEGClassifier with ShallowFBCSPNet as the model
#model = EEGClassifier(
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
#)

model= LogisticRegression()   
#model = svm.SVC()
parameters = {"C": [1, 10, 100]}

#model = RandomForestClassifier()
#parameters = {"n_estimators": [1, 10, 100]}

mean_score, all_true_labels, all_predictions = training_nested_cv_within(model, X, y, parameters, task='classification', groups=groups)
#mean_score, all_true_labels, all_predictions, score_test, most_common_best_param = training_nested_cv_between(model, X, y, parameters = parameters, task = 'classification', nfolds=3, groups=groups)



import numpy as np
from matplotlib import pyplot as plt
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from sklearn.model_selection import GroupKFold

from os.path import join as opj
import torch
import os
import mne
from mne import io
from mne.datasets import sample
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

#____________________________________________________________________________
# Application of cross validation for different models
# Load data


# Get the current working directory
current_directory = os.getcwd()

# Check if the keyword "lustre" is present in the current directory
cuda = "lustre" in current_directory

# If cuda is True and a GPU is available, set up GPU acceleration in PyTorch
# And set bidsroot according to device 
if cuda:
    if torch.cuda.is_available():
        device = torch.device('cuda')  # PyTorch will use the default GPU
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    bidsroot = '/lustre04/scratch/mabus103/epoched data whole/cleaned_epo.fif'
    log_dir=f'/lustre04/scratch/mabus103/ML_for_Pain_Prediction/logs'

elif "mplab" in current_directory:
    device = torch.device('cpu')  # Use CPU if GPU is not available or cuda is False
    #bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/cleaned_epo.fif'
    bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
    log_dir='/home/mplab/Desktop/Mathilda/Project/code/ML_for_Pain_Prediction/logs'
else:
    bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
    log_dir='/home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs'

data_path = opj(bidsroot)
# Load epochs oject
epochs = mne.read_epochs(data_path, preload=True)
# Exclude eog and misc channels
epochs = epochs.pick_types(eeg=True) 

# Set target and label data
epochs_data = epochs.get_data()

# Rescale X to a bigger number
epochs_data = epochs_data * 10e6

labels = epochs.metadata["task"].values  
# Assuming labels are strings, convert them to numerical labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Define the groups (participants) to avoid splitting them across train and test
groups = epochs.metadata["participant_id"].values

n_components = 3  # pick some components

# Define a monte-carlo cross-validation generator (reduce variance):
cv = KFold(n_splits=10, shuffle=True, random_state=42)
pr = np.zeros(len(labels))

clf = make_pipeline(XdawnCovariances(n_components), MDM())

# Split between participants
gkf = GroupKFold(n_splits=3)

for i, (train_idx, test_idx) in enumerate(gkf.split(epochs_data, labels, groups)):
    y_train, y_test = labels[train_idx], labels[test_idx]

    clf.fit(epochs_data[train_idx], y_train)
    pr[test_idx] = clf.predict(epochs_data[test_idx])

# Get unique labels and their indices
unique_labels, indices = np.unique(labels, return_index=True)

# Sort labels based on their indices so the classes appear in the correct order
sorted_labels = [label for _, label in sorted(zip(indices, unique_labels))]

# Transform the sorted labels back to their original string values
sorted_string_labels = label_encoder.inverse_transform(sorted_labels)

# Print the sorted string labels
print("classes corresponding to class numbers in respective order: ", sorted_string_labels)

print(classification_report(labels, pr))
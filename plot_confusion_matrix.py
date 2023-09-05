import pandas as pd
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
#import pandas as pd
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

# Load the CSV file that contains the True and Predicted Labels
csv_file = f"resultsshallowFBCSPNetClassification/between.csv"  
df = pd.read_csv(csv_file)

# Get unique labels from the "True Label" column
unique_labels = df["True Label"].unique()

# Compute the confusion matrix
cm = confusion_matrix(df["True Label"], df["Predicted Label"])

# Create a mapping from the index in unique_labels to the actual label
label_mapping = {i: label for i, label in enumerate(unique_labels)}

# Plot confusion matrix with correct label mapping
fig, ax = plt.subplots(1)
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.set(title="Confusion matrix")

# Set x and y axis tick labels based on unique labels from CSV using label_mapping
tick_marks = np.arange(len(unique_labels))
plt.xticks(tick_marks, [label_mapping[i] for i in range(len(unique_labels))], rotation=45)
plt.yticks(tick_marks, [label_mapping[i] for i in range(len(unique_labels))])

fig.colorbar(im)
fig.tight_layout(pad=1.5)
ax.set(ylabel="True label", xlabel="Predicted label")

"""# Save the confusion matrix plot as an image file
output_dir = f"images/confusion_matrix/resultsshallowFBCSPNetClassification"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_file = os.path.join(output_dir, f"{part}.png")
plt.savefig(output_file)"""

# Show the plot
plt.show()
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
csv_file = f"/home/mathilda/MITACS/Project/CC/results/resultsshallowFBCSPNetClassification/within.csv"  
df = pd.read_csv(csv_file)

"""# Get unique labels from the "True Label" column
unique_labels = df["True Label"].unique()

# Compute the confusion matrix
cm = confusion_matrix(df["True Label"], df["Predicted Label"])

# Create a mapping from the index in unique_labels to the actual label
label_mapping = {i: label for i, label in unique_labels}

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
ax.set(ylabel="True label", xlabel="Predicted label")"""

"""# Save the confusion matrix plot as an image file
output_dir = f"images/confusion_matrix/resultsshallowFBCSPNetClassification"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_file = os.path.join(output_dir, f"{part}.png")
plt.savefig(output_file)"""

# Show the plot
#plt.show()

model_name = "shallowFBCSPNet"
part = "within"
# Load data from the CSV file
data = df
true_labels = data['True Label']
predicted_labels = data['Predicted Label']

# Get unique class labels
classes = sorted(list(set(true_labels) | set(predicted_labels)))
classes2 = ["auditory", "rest", "thermal"]
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
plt.xticks(tick_marks, classes2, rotation=45)
plt.yticks(tick_marks, classes2)
for i in range(len(classes)):
    for j in range(len(classes)):
        text = "{:.2f}".format(cm_normalized[i, j])
        plt.text(j, i, text, ha='center', va='center', color='white')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout(pad=1.5)


#legend_labels = {
#        0: 'auditory',
 #       1: 'rest',
  #      2: 'thermal'
   # }

#legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f"{label}: {text}", markersize=10, markerfacecolor='b') for label, text in legend_labels.items()]
#plt.legend(handles=legend_elements, title="Legend")

#plt.show()

# Save the confusion matrix plot as an image file
output_dir = f"images/confusion_matrix{model_name}"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_file = os.path.join(output_dir, f"{part}.png")
plt.savefig(output_file)
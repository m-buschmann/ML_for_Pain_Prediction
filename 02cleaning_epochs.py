#!/usr/bin/env python

import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from autoreject import AutoReject


# Set root directory
#bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives'
bidsroot = '/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives'

# Create cleaned epochs dir
pdir = opj(bidsroot, 'cleaned_epo.fif')

# Get participants
part = sorted([s for s in os.listdir(bidsroot) if "sub-" in s])

# Output folder for csv file and epochs opbject
derivpath = opj(bidsroot)

# Create a data frame path to collect stats
excluded_csv_path = opj(bidsroot, 'cleaned epochs', 'excluded_epochs_autoreject.csv')

# Create a data frame to collect stats if not already created
if not os.path.exists(excluded_csv_path):
    excluded = pd.DataFrame(columns=["Participant", "Task", "Excluded epochs"])
else:
    excluded = pd.read_csv(excluded_csv_path)


# Loop participants (3 sub are very bad, excluded)
part = [p for p in part if p not in ["sub-015", "sub-018", "sub-029"]]


# Loop participants
for p in part:
    print(p)
    all_epochs = None

    # Loop tasks
    for task in ["thermalrate", "auditoryrate", "thermal", "auditory", "rest"]:

        # Load raw file
        raw = mne.io.read_raw_fif(
            opj(bidsroot,p, "eeg", p + "_task-" + task + "_cleanedeeg-raw.fif"),
            preload=True,
        )    

        # Make windows
        epochs = mne.make_fixed_length_epochs(raw, duration=4, overlap=0, preload=True)

        # run autoreject
        ar = AutoReject(
            n_jobs = -1,
            n_interpolate = [0],
            random_state=42)

        #ar.fit(epochs)

        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True) #fit transform

        # Get indices of rejected epochs #also get ratio of rejected epochs
        rejected_epochs = len(np.where(reject_log.bad_epochs)[0])

        #add row to statistics
        excluded.loc[len(excluded)] = [p, task, rejected_epochs]

        if task == "thermalrate" or  "thermal":
            dim = -2,
        elif task == "auditoryrate" or "auditory":
            dim = -3,
        elif task == "rest":
            dim = "nan"

        # Get the average rating for each epoch
        intensity = [np.mean(e[-1, :]) for e in epochs_clean]
        rating = [np.mean(e[dim, :]) for e in epochs_clean]

        # Get the average rating difference in each epoch
        diff_rate = [np.max(e[-2, :]) - np.min(e[-2, :]) for e in epochs_clean]
        diff_stim = [np.max(e[-1, :]) - np.min(e[-1, :]) for e in epochs_clean]

        # Add metadata
        meta_data = pd.DataFrame(
            {
                "participant_id": p,
                "task": task,
                "rating": rating,
                "epoch_num": np.arange(len(epochs_clean)),
                "intensity": intensity,
                "diff_rate": diff_rate,
                "diff_intensity": diff_stim,
                "reject_prop": 1 - (len(epochs_clean) / len(epochs)),
            }
        )
        
        # set metadata
        epochs_clean.metadata = meta_data
        epochs_clean.resample(250)

        if all_epochs is None:
            all_epochs = epochs_clean
        else:
            all_epochs = mne.concatenate_epochs([all_epochs, epochs_clean])

        # Save statistics
        excluded.to_csv(opj(derivpath,'cleaned epochs2', "excluded_epochs_autoreject.csv"), index=False)
        
    # Save the epochs object after each participant
    cleaned_epo_path = opj(derivpath,'cleaned epochs2',  p +'_cleaned_epo.fif')
    all_epochs.save(cleaned_epo_path, overwrite=True)
        
    # Explicitly delete objects to release memory
    del raw, epochs, epochs_clean, reject_log, meta_data ,all_epochs




# Concatenate the cleaned epochs of all participants
all_epochs = None
bidsroot ='/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs2'
part = sorted([s for s in os.listdir(bidsroot) if "sub-" in s])
derivpath = opj(bidsroot)

# Create a generator function to yield the EpochsArray objects
def epochs_generator(participants, bidsroot):
    for p in participants:
        cleaned_epo_path = opj(bidsroot, p)
        epochs_clean = mne.read_epochs(cleaned_epo_path, preload=True)
        yield epochs_clean

# Get the list of participants
part = sorted([s for s in os.listdir(bidsroot) if "sub-" in s])


# Concatenate the cleaned epochs of all participants using the generator
all_epochs = None
for epochs_clean in epochs_generator(part, bidsroot):
    if all_epochs is None:
        all_epochs = epochs_clean
    else:
        all_epochs = mne.concatenate_epochs([all_epochs, epochs_clean])

# Resample the final concatenated epochs to the desired rate
all_epochs.resample(250)

# Save the final all_epochs object
all_epochs.save(opj(bidsroot, 'cleaned_epo.fif'), overwrite=True)

"""
# Concatenate the two big epoch objects -> doesnt work, memory?
all_epochs = None
bidsroot ='/home/mplab/Desktop/Mathilda/Project/eeg_pain_v2/derivatives/cleaned epochs'
part = sorted([s for s in os.listdir(bidsroot) if "sub" in s])
derivpath = opj(bidsroot)

part = [p for p in part if p in ["first_29_sub_cleaned_epo.fif", "second_sub_cleaned_epo.fif"]]

# Create a generator function to yield the EpochsArray objects
def epochs_generator(participants, bidsroot):
    for p in participants:
        cleaned_epo_path = opj(bidsroot, p)
        epochs_clean = mne.read_epochs(cleaned_epo_path, preload=True)
        yield epochs_clean

# Concatenate the cleaned epochs of all participants using the generator
all_epochs = None
for epochs_clean in epochs_generator(part, bidsroot):
    if all_epochs is None:
        all_epochs = epochs_clean
    else:
        all_epochs = mne.concatenate_epochs([all_epochs, epochs_clean])

# Resample the final concatenated epochs to the desired rate
all_epochs.resample(250)

# Save the final all_epochs object
all_epochs.save(opj(bidsroot, 'cleaned_epo.fif'), overwrite=True)"""
import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from autoreject import AutoReject

#TODO: hardcode to keep some bad channels
#TODO: run on all part

# Set root directory
bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives'

# Output folder for csv file and epochs opbject
derivpath = opj(bidsroot)

# epochs object 
all_epochs = None

# Get participants
part = sorted([s for s in os.listdir(bidsroot) if "sub-" in s])

# Create a data frame path to collect stats
excluded_csv_path = opj(bidsroot, 'excluded_epochs_autoreject.csv')

# Create a data frame to collect stats 
excluded = pd.DataFrame(columns=["Participant", "task", "epochs"])


# Loop participants (3 sub are very bad, excluded)
part = [p for p in part if p not in ["sub-015", "sub-018", "sub-029"]]
#part = [p for p in part if p in ["sub-003", "sub-004"]]

# Loop participants
for p in part:
    print(p)

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
            n_interpolate = [0],
            random_state=42)
        
        ar.fit(epochs)
        epochs_clean, reject_log = ar.transform(epochs, return_log=True)

        # Get indices of rejected epochs
        rejected_epochs = len(np.where(reject_log.bad_epochs)[0])

        #add row to statistics
        excluded.loc[len(excluded)] = [p, task, rejected_epochs]

        # Get the average rating for each epoch
        intensity = [np.mean(e[-1, :]) for e in epochs_clean]
        rating = [np.mean(e[-2, :]) for e in epochs_clean]

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

        # Concatenate the clean epochs of the current participant to all_epochs
        if all_epochs is None:
            all_epochs = epochs_clean
        else:
            all_epochs = mne.concatenate_epochs([all_epochs, epochs_clean])

    # Save statistics
    excluded.to_csv(opj(derivpath, "excluded_epochs_autoreject.csv"))

# Resample
all_epochs.resample(250)

#save the final epochs object
all_epochs.save(opj(derivpath,'2_cleaned_epo.fif'), overwrite=True)
# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2023-03-16 20:51:59
# @Last Modified by:   Your name
# @Last Modified time: 2023-07-26 10:36:58
"""author: mpcoll."""

import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
import scipy
from mne.report import Report
from mne.preprocessing import ICA,  corrmap, create_ecg_epochs, create_eog_epochs
from autoreject import Ransac
from mne_icalabel import label_components
from meegkit import dss

# %matplotlib inline
# %matplotlib qt5

# TODO : @MB add different rows for each task in stats frame DONE
# TODO: @ MB add additional figures to report DONE
# TODO @MB investiate files with lots of bad channels

###############################
# Parameters
###############################
bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2'
# Get participants
#part = sorted([s for s in os.listdir(bidsroot) if "sub-" in s])

# Output folder
derivpath = opj(bidsroot, "derivatives")

# Loop participants (sub-015 is very bad, excluded before any processing)
#part = [p for p in part if p in ["sub-003", "sub-004"]]


p = "sub-042"
# Loop participants
#for p in part:
print(p)
# Create participant output dir
pdir = opj(derivpath, p, "eeg")
if not os.path.exists(pdir):
    os.makedirs(pdir)

# Loop tasks
#for task in ["thermalrate", "auditoryrate", "thermal", "auditory", "rest"]:
for task in ["rest"]:

    # Load raw file
    raw = mne.io.read_raw_brainvision(
        opj(bidsroot, p, "eeg", p + "_task-" + task + "_eeg.vhdr"),
        eog=["VEOG", "HEOG"],  # Specify eog channels
        preload=True,
    )

    # Set channel positions
    raw = raw.set_montage("easycap-M1")

    #picks = mne.pick_channels(raw.ch_names, ["FCz", "Cz"])
    #raw.copy().set_eeg_reference("average").plot(order=picks, n_channels=len(picks))

    raw.copy().set_eeg_reference("average").plot(start=10, duration=10, n_channels = 15)
    print(raw.info["bads"])
    # Add manually flagged bad channels
    chan = pd.read_csv(
        opj(bidsroot, p, "eeg", p + "_task-" + task + "_channels.tsv"), sep="\t"
    )
    print(raw.info["bads"])
    raw.info["bads"] = chan[chan["status"] == "bad"]["name"].tolist()
    print(raw.info["bads"])

    
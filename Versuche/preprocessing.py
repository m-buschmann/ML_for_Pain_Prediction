import os
import mne
import autoreject
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
import matplotlib.pyplot as plt
from autoreject import Ransac
from autoreject import (AutoReject, set_matplotlib_defaults)
import numpy as np 
import pandas as pd
from mne_icalabel import label_components
import re

#initialize an empty dataframe to store the results for all subjects
df_results = pd.DataFrame(columns=["Subject", "Bad_Channels", "Dropped_Epochs"])
bad_channels_list = []
dropped_epochs_list = []

#configure setting
mne.use_log_level("warning")
mne.set_config("MNE_USE_CUDA", "True")
print(mne.get_config("MNE_USE_CUDA"))
"""
#get a list of all subfolders (subject IDs) in the data folder
subject_folders = [subfolder for subfolder in os.listdir(sample_data_folder) if os.path.isdir(os.path.join(data_folder, subfolder))]

#loop through each subject folder
for subject_id in subject_folders:
    subject_path = os.path.join(data_folder, subject_id)
"""

#load data
subject_id = "sub-020"
sample_data_folder = '/home/mathilda/MITACS/Project/eeg_pain_data'
eeg_file = os.path.join(
    sample_data_folder, subject_id, "eeg", subject_id + "_task-painaudio_eeg.vhdr"
)
raw = mne.io.read_raw_brainvision(eeg_file, preload=True)

#set up report
report = mne.Report(title="report data preprocessing")
report.add_raw(raw=raw, title="Raw at start", psd=True)  #set the eog channels here

#set montage
dig_montage = mne.channels.make_standard_montage("easycap-M1")
raw.drop_channels([ch for ch in raw.ch_names
                   if ch not in dig_montage.ch_names]) #? 
raw.set_montage(dig_montage)
#raw.set_montage("easycap-M1")


#_____________________________________________________________________________
#filter data

raw.notch_filter((60, 200, 60), method='spectrum_fit')

# insert bandpas filter here 0.3 - 100
#add filtered raw to report
report.add_raw(raw=raw, title="Raw after notch filter", psd=True)

#make epochs for ransac
epochs = mne.make_fixed_length_epochs(raw.copy(), duration=3, preload=True)#add more filter like [1,60], set to average, drop bad epochs 200e-6

#fit RANSAC algorithm to epoched data
ransac = Ransac()
ransac.fit(epochs)

#get the list of bad channels
bad_channels = ransac.bad_chs_
bad_channels_list = bad_channels

#mark bad channels as bad in the raw data
raw.info['bads'] += bad_channels[:]

#print the list of bad channels
print("Bad Channels:", bad_channels)

#add montage to report
fig_montage = raw.plot_sensors(show_names=True)
report.add_figure(fig_montage, "Montage")

# _________________________________________________________________
# Epoch according to condition to remove breaks and pre/post

#set path to file
events = pd.read_csv(
    os.path.join(sample_data_folder,subject_id, "eeg", subject_id + "_task-painaudio_events.tsv"),
    sep="\t"
)

# Epochs are 500 s
events["offset"] = events["onset"] + raw.info["sfreq"] * 500

# Keep only start of trial
events = events[
    events.condition.isin(
        [
            "rest_start",
            "thermal_start",
            "auditory_start",
            "thermalrate_start",
            "auditoryrate_start",
        ]
    )
]

events_dict = {
    "rest_start": 1,
    "thermal_start": 2,
    "auditory_start": 3,
    "thermalrate_start": 4,
    "auditoryrate_start": 5,
}

events_c = pd.DataFrame(columns=["cue_num", "empty", "sample"])
events_c["cue_num"] = [events_dict[s] for s in events.condition]
events_c["sample"] = list(events["onset"])
events_c["empty"] = 0
events_epochs = np.asarray(events_c[["sample", "empty", "cue_num"]])

# Create epochs
epochs_good = mne.Epochs(
    raw,
    events=events_epochs,
    event_id=events_dict,
    tmin=0,
    tmax=500,
    reject=None,
    baseline=None,
    preload=True,
)

# Make a new raw with only times of interest for ICA
raw_cont = mne.io.RawArray(np.hstack(epochs_good.get_data()), raw.info)

#___________________________________________________________________________________
#work on copy: filtering and ICA
#order different than what was discussed

#reference to average
raw_cont.set_eeg_reference(ref_channels='average')#make copy here

#filter 1-100 Hz
raw_cont.filter(l_freq=1, h_freq=100)

epochs = mne.make_fixed_length_epochs(raw_cont, duration=3, preload=True)

#drop epochs peak-to-peak
epochs.drop_bad(reject=dict(eeg=500e-6))
fig = epochs.plot_drop_log()
report.add_figure(fig, "Dropped epochs due to peak-to-peak amplitude")
dropped_epochs_list.append(list(epochs.drop_log[:]))

#bad channels list
#bad_channels = epochs.info['bads']

#add bad channels to report before interpolation
fig = epochs.plot(picks=bad_channels, scalings='auto', show=True, block=True)
report.add_figure(fig, "bad channels before RANSAC")
my_html = bad_channels
report.add_html(title="List of bad channels before interpolation", html=my_html)

#apply RANSAC to the filtered epochs, identify and mark bad channels after ICA!!! just interpolation here
ransac.transform(epochs)
epochs.interpolate_bads(reset_bads=True) #not sure if it worked correctly

#bad channels list 
bad_channels = epochs.info['bads']
print(bad_channels)
bad_channels_list.append(bad_channels[:]) #should be empty after ransac

#update bad channels in raw object
#raw_cont.info['bads'] = raw_info['bads'] #alternatively, just to make sure ica has correct shape
raw_cont.info['bads'] = bad_channels[:]

#add bad channels to report after interpolation
my_html = bad_channels
report.add_html(title="List of bad channels after interpolation", html=my_html)

#new raw object from epochs
raw_cont = mne.io.RawArray(np.hstack(epochs.get_data()), raw_cont.info)

#add montage to report
fig_montage = raw_cont.plot_sensors(show_names=True) 
report.add_figure(fig_montage, "Channels after RANSAC")

#plot in report
report.add_raw(raw=raw_cont, title="Raw after averaging, dropping epochs, interpolation", psd=True)

#set up ica model and fit
ica = ICA(
    n_components=20, #just to reduce computing time, set to None later
    method="infomax",
    random_state=97,
    fit_params=dict(extended=True),
)
ica.fit(raw_cont, decim=4) #here fit on epochs

#get labels
ic_labels = label_components(raw_cont, ica, method="iclabel")
print(ic_labels)

#list of ics that are not brain
ic_indices = [i for i, (proba, label) in enumerate(zip(ic_labels["y_pred_proba"], ic_labels["labels"])) if proba > 0.8 and label != "brain"] #keep 'other', just flag the single ics

#add bad ics to report
report.add_ica(
    ica=ica,
    title="ICA cleaning",
    picks=ic_indices, 
    inst=raw_cont,
    n_jobs=None,
)

#eclude these ics
ica.exclude = ic_indices

#apply the exclusion to the raw object
ica.apply(raw_cont, exclude=ica.exclude)

#_______________________________________________________________________________________________________
#apply to original data
# just use raw_cont

#also drop bad epochs?

#interpolate bad channels in the raw data
raw.interpolate_bads() #? without doing this at first, raw does not have correct shape for ica

#stop here, save raw to derivative file (raw, dataframe and report for participant)
#apply ICA to the original data
ica.apply(raw, exclude=ica.exclude)

if len(ica.exclude) > 0:
    print("ICA applied successfully. Excluded components:", ica.exclude)
else:
    print("ICA applied successfully. No components were excluded.")

#plot in report
report.add_raw(raw=raw, title="Raw after ICA", psd=True) #why still powerline noise? bc here no epochs dropped?

#create epochs using windowing
window_length = 4.0  
window_overlap = 0.0  #?

windows = mne.make_fixed_length_epochs(raw, duration=window_length, overlap=window_overlap, preload=True) #??? is this windowing

#interpolate bad segments in the epochs data using AutoReject
ar = AutoReject(n_interpolate=[1,2],random_state=11, n_jobs=1, verbose=True)
ar.fit(windows)
epochs_ar, reject_log = ar.transform(windows, return_log=True)

#average reference
windows.set_eeg_reference(ref_channels='average')

#interpolate bad channels in the raw data
#raw.interpolate_bads()

#add the cleaned epochs to the report
report.add_epochs(windows, title="Cleaned Epochs") 
#save report to a file
report_file = os.path.join('/home/mathilda/MITACS/Project', "filtering_report.html")
report.save(report_file, overwrite=True)

#clean up the dropped epochs list to store only dropped epochs
clean_dropped_epochs_list = []
for i, subject_dropped_epochs in enumerate(dropped_epochs_list):
    subject_clean_dropped_epochs = [list(epoch_log) for epoch_log in subject_dropped_epochs if epoch_log != ()]
    clean_dropped_epochs_list.append((i, subject_clean_dropped_epochs))

#append the results to the DataFrame
df_temp = pd.DataFrame(
    {
        "Subject": [subject_id],
        "Bad_Channels": [bad_channels_list],
        "Dropped_Epochs": [clean_dropped_epochs_list],
    }
)
df_results = pd.concat([df_results, df_temp], ignore_index=True)

#save the results dataframe to a CSV file
df_results.to_csv("/home/mathilda/MITACS/Project/Results_df.csv", index=False)
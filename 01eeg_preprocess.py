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
from mne.preprocessing import ICA
from autoreject import Ransac
from mne_icalabel import label_components
from meegkit import dss

# %matplotlib inline
# %matplotlib qt5

###############################
# Parameters
###############################
bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2'

# Get participants
part = sorted([s for s in os.listdir(bidsroot) if "sub-" in s])

# Output folder
derivpath = opj(bidsroot, "derivatives")

# Loop participants (sub-015 is very bad, excluded before any processing)
part = [p for p in part if p not in ["sub-015"]]

# Create a data frame to collect stats
stats_frame = pd.DataFrame(columns=["Participant", "Subtask", "n_bad_chans", "n_bads_ica"])

# Loop participants
for p in part:
    print(p)
    # Create participant output dir
    pdir = opj(derivpath, p, "eeg")
    if not os.path.exists(pdir):
        os.makedirs(pdir)

    # Loop tasks
    for task in ["thermalrate", "auditoryrate", "thermal", "auditory", "rest"]:
        # Initialize report
        report = Report(
            verbose=False, subject=p, title="EEG report for part " + p + " task " + task
        )

        # Load raw file
        raw = mne.io.read_raw_brainvision(
            opj(bidsroot, p, "eeg", p + "_task-" + task + "_eeg.vhdr"),
            eog=["VEOG", "HEOG"],  # Specify eog channels
            preload=True,
        )
        report.add_raw(raw=raw.copy().filter(l_freq=0.5, h_freq=100).set_eeg_reference("average"), title="Raw at start", psd=True) #? filter 1-100?

        # Set channel positions
        raw = raw.set_montage("easycap-M1")

        # Add manually flagged bad channels
        chan = pd.read_csv(
            opj(bidsroot, p, "eeg", p + "_task-" + task + "_channels.tsv"), sep="\t"
        )

        raw.info["bads"] = chan[chan["status"] == "bad"]["name"].tolist()

        # _________________________________________________________________
        # Identify additional bad channels using ransac

        # Epochs for ransac (filtered, 2s epochs with average reference and very bad epochs removed)
        epochs = (
            mne.make_fixed_length_epochs(raw.copy().filter(1, 60), duration=2)
            .load_data()
            .set_eeg_reference("average")
            .drop_bad(reject=dict(eeg=200e-6))
        )

        ransac = Ransac(verbose=True, picks="eeg", n_jobs=-1)
        ransac.fit(epochs)

        raw.info["bads"] += ransac.bad_chs_
        #remove some of the channels from the bad list because they seem to be okay
        if p == "sub-012" and task == "rest":
            channels_to_remove = ["F5", "FCz", "P1", "CP4"]  # Add the channel names you want to remove
            raw.info["bads"] = [ch for ch in raw.info["bads"] if ch not in channels_to_remove]
        elif p == "sub-040" and task == "rest":
            channels_to_remove = ["AF3", "F5", "FT9",  "FT7", "FT8",  "FT10"]  # Add the channel names you want to remove
            raw.info["bads"] = [ch for ch in raw.info["bads"] if ch not in channels_to_remove]
        elif p == "sub-042" and task == "rest":
            channels_to_remove = ["C6","FT9"]  # Add the channel names you want to remove
            raw.info["bads"] = [ch for ch in raw.info["bads"] if ch not in channels_to_remove]


        # plot channels with bads in red
        fig = raw.plot_sensors(show_names=True, show = False)
        report.add_figure(fig, "Sensor positions (bad in red)")

        # Average reference
        raw = raw.set_eeg_reference("average", projection=False)

        # Clear memory
        epochs = None
        raw_f = None

        # Collect rejection statistics in frame
        n_bad_chans = len(raw.info["bads"])

        # _________________________________________________________________
        # Filter and zapline
        # Plot raw spectrum
        plt_psd = raw.copy().set_eeg_reference("average").plot_psd(fmax=100, show=False)
        report.add_figure(plt_psd, "Raw spectrum")

        # Zapline iteratively to remove line noise
        # Keep only good eeg channels

        # Good eeg idx
        # TODO @MB Double check this step
        """bad_idx = [
            i
            for i in range(len(raw.info["ch_names"]))
            if raw.info["ch_names"][i] in raw.info["bads"]
        ]"""
        # EEG idx
        eeg_idx = mne.channel_indices_by_type(raw.info)["eeg"]
        bad_names = raw.info["bads"]
        bad_idx = [eeg_idx[raw.ch_names.index(name)] for name in bad_names if name in raw.ch_names]

        # Idx to keep (union of eeg and not bad)
        keep_idx = [i for i in eeg_idx if i not in bad_idx]

        # Get data
        x_unfilt = raw.copy().get_data()[keep_idx, :]

        # Zapline (dss needs time first so transpose)
        x_filt, _ = dss.dss_line_iter(
            np.swapaxes(x_unfilt, 1, 0), 60, raw.info["sfreq"]
        )

        # REplace data with filtered one
        x_final = raw.copy().get_data()
        x_final[keep_idx, :] = np.swapaxes(x_filt, 1, 0)

        # Load filtered data back into mne
        raw = mne.io.RawArray(x_final, raw.info)

        # Final Bandpass filter
        raw = raw.filter(0.3, 100)

        # Plot spectrum after line noise removal/highpass
        plt_psd = raw.copy().set_eeg_reference("average").plot_psd(fmax=100, show=False)
        report.add_figure(plt_psd, "Spectrum, line noise removed + bandpass")

        # _________________________________________________________________

        # Epochs for ICA with additional high pass filter (as recommended by MNE)
        epochs = mne.make_fixed_length_epochs(
            raw.copy().load_data().filter(1, 100), duration=2
        ).load_data()
        # Remove very bad epochs for ICA
        epochs.drop_bad(reject=dict(eeg=400e-6))

        # ICA
        ica = ICA(
            n_components=None,
            method="infomax",
            fit_params=dict(extended=True),
            random_state=23,
        )
        # Fit ICA
        ica.fit(epochs, decim=4)

        # Run ICA labels
        ica_labels = label_components(epochs, ica, method="iclabel")

        # Remove components labveled as bad with > X probability
        remove = [
            1
            if ic
            in [
                "channel noise",
                "eye blink",
                "muscle artifact",
                "line noise",
                "heart beat",
                "eye movement",
            ]
            and prob > 0.90
            else 0
            for ic, prob in zip(ica_labels["labels"], ica_labels["y_pred_proba"])
        ]
        ica.exclude = list(np.argwhere(remove).flatten())

        # Collect number in frame and report
        n_bads_ica = len(ica.exclude)

        # Add ICA labels to report
        report.add_html(pd.DataFrame(ica_labels).to_html(), "ICA labels")

        # Add ICA components to report
        report.add_ica(ica, title="ICA components", inst=epochs)

        # Apply ICA to the data
        ica.apply(raw, exclude=ica.exclude)

        my_html = raw.info["bads"]
        report.add_html(title="List of bad channels before interpolation", html=my_html)
        
        # Interpolate bad channels after ICA
        raw = raw.interpolate_bads(reset_bads=True)

        # ______________________________________________________________________
        # Add cleaned raw to the report
        report.add_raw(raw, title="Cleaned raw data", psd = True)

        # Save cleaned data
        raw.save(opj(pdir, p + "_task-" + task + "_cleanedeeg-raw.fif"), overwrite=True)
        
        #add the cleaned epochs to the report
        report.add_epochs(epochs, title="Cleaned Epochs")
        #  _____________________________________________________________________
        # Save report
        report.save(
            opj(pdir, p + "_task-" + task + "_preprocess_report.html"),
            open_browser=False,
            overwrite=True,
        )
        #add row to statistics
        stats_frame.loc[len(stats_frame)] = [p, task, n_bad_chans, n_bads_ica]

    # Save statistics
    stats_frame.to_csv(opj(derivpath, "preprocess_stats.csv"))

import os
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

sample_data_folder = '/home/mathilda/MITACS/Project/eeg_pain_data'
eeg_file = os.path.join(
    sample_data_folder,"sub-020", "eeg", "sub-020_task-painaudio_eeg.vhdr"
)
raw = mne.io.read_raw_brainvision(eeg_file, preload=True)


"""subfolders = [f.path for f in os.scandir(sample_data_folder) if f.is_dir() and f.name.startswith('sub-')]

for subfolder in subfolders:
    eeg_folder = os.path.join(subfolder, 'eeg')

    # Find the vhdr file in the eeg folder
    vhdr_file = [f for f in os.listdir(eeg_folder) if f.endswith('.vhdr')][0]
    eeg_file = os.path.join(eeg_folder, vhdr_file)

    # Load the EEG data
    raw_all = mne.io.read_raw_brainvision(eeg_file, preload=True)

    raw_all.plot(duration=5, n_channels=30, block= True)"""



print(raw)
print(raw.info)

#raw data plot
raw.plot(start=10, duration=5, n_channels=20)

#psd plot
fig = raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", show=True)









"""# Get the list of channel names
channel_names = raw.info['ch_names']

# Find EOG channels
eog_channels = [ch_name for ch_name in channel_names if 'EOG' in ch_name]

# Print the number of EOG channels and their names
print("Number of EOG channels:", len(eog_channels))
print("EOG channel names:", eog_channels)"""


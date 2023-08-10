import mne
from os.path import join as opj

#TODO: hardcode to keep some bad channels
#TODO: 

bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives'

# Load raw file
epochs = mne.read_epochs(
    opj(bidsroot, "cleaned epochs","single_sub_cleaned_epochs", "sub_3_to_11_cleaned_epo.fif"),
    preload=True,
)

"""filtered_epochs = epochs[epochs.metadata["participant_id"] == "sub-007"]
filtered_epochs = filtered_epochs[filtered_epochs.metadata["task"] == "thermalrate"]

print(filtered_epochs.metadata)"""

print(epochs.metadata)

# Check for duplicates in metadata
duplicated_rows = epochs.metadata.duplicated()
if any(duplicated_rows):
    print("Duplicate rows found!")
    print(epochs.metadata[duplicated_rows])
else:
    print("No duplicate rows found.")
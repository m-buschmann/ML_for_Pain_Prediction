from sklearn.model_selection import StratifiedGroupKFold
import mne
from os.path import join as opj
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error

# Directory
bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_11_cleaned_epo.fif'
data_path = opj(bidsroot)
# Load epochs oject
epochs = mne.read_epochs(data_path, preload=True)

# Set target and label data
X = epochs.get_data()
#y = epochs.metadata["rating"].values # .value correct? else, missing epochs are a problem
y = epochs.metadata["task"].values


# Define the groups (participants) to avoid splitting them across train and test
groups = epochs.metadata["participant_id"]

# Create a pipeline for preprocessing and classification
pipe = make_pipeline(
    mne.decoding.Scaler(scalings='mean'), # Scale the data
    mne.decoding.Vectorizer(), # Vectorize the data
    LogisticRegression() # Logistic Regression Classifier
)

# Initialize StratifiedGroupKFold with the desired number of folds
sgkf = StratifiedGroupKFold(n_splits=3)

# Initialize an array to store accuracy scores for each fold
accuracy_scores = []

# Cross-validation loop
for i, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):

    # Split data into training and test sets based on the current fold indices
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model on the training data
    pipe.fit(X_train, y_train)
    # Predict on the test set
    y_pred = pipe.predict(X_test)

    # Calculate accuracy using accuracy_score for the current fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print("Accuracy in fold", i, ":", accuracy)

# Calculate the mean accuracy across all folds
mean_accuracy = np.mean(accuracy_scores)

print("Mean Accuracy: {:.2f}%".format(mean_accuracy * 100))

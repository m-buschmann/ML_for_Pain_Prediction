import mne
from os.path import join as opj
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error

# Directory
bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
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
    LogisticRegression(C=10) # Logistic Regression Classifier
)

# Initialize GroupKFold with the desired number of folds
gkf = GroupKFold(n_splits=3)

# Initialize an array to store accuracy/mse scores for each fold
accuracy_scores = []
#mse_scores = []

# Initialize arrays to store true labels and predictions for each fold
all_true_labels = []
all_predictions = []


# Cross-validation loop
for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

    # Split data into training and test sets based on the current fold indices
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Shuffle within the training set
    train_shuffle_indices = np.random.permutation(len(y_train))
    X_train, y_train = X_train[train_shuffle_indices], y_train[train_shuffle_indices]

    # Fit the model on the training data
    pipe.fit(X_train, y_train)
    # Predict on the test set
    y_pred = pipe.predict(X_test)

    """# Calculate mean squared error (MSE) for the current fold
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    print("Mean Squared Error in fold", i, ":", mse)"""

    # Calculate accuracy using accuracy_score for the current fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print("Accuracy in fold", i, ":", accuracy)

    # Append true labels and predictions to the corresponding lists
    all_true_labels.extend(y_test)
    all_predictions.extend(y_pred)

"""# Calculate the mean mean squared error across all folds
mean_mse = np.mean(mse_scores)

print("Mean Mean Squared Error: {:.2f}".format(mean_mse))
"""

# Calculate the mean accuracy across all folds
mean_accuracy = np.mean(accuracy_scores)

print("Mean Accuracy: {:.2f}%".format(mean_accuracy * 100))

# Output the first 10 elements of true labels and predictions
print("True Labels (First 10 elements):", all_true_labels[:50])
print("Predictions (First 10 elements):", all_predictions[:50])

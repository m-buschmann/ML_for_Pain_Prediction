import mne
from os.path import join as opj
import numpy as np
from sklearn import datasets, linear_model, svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import Counter

def within_participant_DL(model, X, y, task='classification', groups=None):
    # Initialize an array to store accuracy scores for each participant
    participant_scores = []
    # Initialize arrays to store true labels and predictions for each fold
    all_true_labels = []
    all_predictions = []
    # Create a pipeline for preprocessing and classification
    pipe = make_pipeline(
        mne.decoding.Scaler(scalings='mean'),  # Scale the data
        mne.decoding.Vectorizer(),  # Vectorize the data
        model
    )

    # Get unique participant IDs
    unique_participants = np.unique(groups)

    # Loop over each participant
    for participant in unique_participants:

        # Get the data indices for the current participant
        participant_indices = np.where(groups == participant)[0]
        print(participant_indices)
        print(groups)
        # Split participant data into training and testing using train_test_split
        train_idx, test_idx = train_test_split(participant_indices, test_size=0.2, random_state=42)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit the model on the training data
        pipe.fit(X_train, y_train)
        # Predict on the test set
        y_pred = pipe.predict(X_test)

        # Append true labels and predictions to the corresponding lists
        all_true_labels.extend(y_test) #also for test set?
        all_predictions.extend(y_pred)
            
        if task == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            participant_scores.append(mse)
            print("Participant", participant, "MSE:", mse)

        if task == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            participant_scores.append(accuracy)
            print("Participant", participant, "Accuracy:", accuracy)

    # Output the first 10 elements of true labels and predictions
    print("True Labels (First 10 elements):", all_true_labels[:10])
    print("Predictions (First 10 elements):", all_predictions[:10])

    # Calculate the mean accuracy across all participants
    mean_score = np.mean(participant_scores)
    print("Mean Accuracy/MSE across all participants: {:.2f}".format(mean_score))

    return mean_score, all_true_labels, all_predictions

def within_participant_nested_cv(model, X, y, parameters, task = 'regression', nfolds=5, groups=None):
    # Initialize arrays to store true labels and predictions for each fold
    all_true_labels = []
    all_predictions = []   

    # Store scores from outer loop
    score_test = [] 
    
    # Initialize dictionaries to store best parameters and their occurrences
    best_params_counts = Counter()
    best_params_per_fold = {}

    # Create a pipeline for preprocessing
    preprocessing_pipe = make_pipeline(
        mne.decoding.Scaler(scalings='mean'), # Scale the data
        mne.decoding.Vectorizer() # Vectorize the data
    )
    X = preprocessing_pipe.fit_transform(X)

    #inner_group = [] 

    # Outer cross-validation
    # Initialize GroupKFold with the desired number of folds
        # Get unique participant IDs
    unique_participants = np.unique(groups)

    # Loop over each participant
    for participant in unique_participants:
        # Get the data indices for the current participant
        participant_indices = np.where(groups == participant)[0]

        # Split participant data into training and testing using train_test_split
        train_idx, test_idx = train_test_split(participant_indices, test_size=1/nfolds)
        
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        inner_scores = []
        # inner cross-validation
        inner = KFold(2) #increase with more data
        for train_index_inner, test_index_inner in inner.split(X_train_outer, y_train_outer):
            # split the training data of outer CV
            X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
            y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

            # fit regressor to training data of inner CV
            clf = GridSearchCV(model, parameters)
            clf.fit(X_train_inner, y_train_inner)
            inner_scores.append(clf.score(X_test_inner, y_test_inner))
            
            # Store best parameters for each fold
            best_params_fold = clf.best_params_
            best_params_counts.update([str(best_params_fold)])  # Convert to string for dictionary key
            best_params_per_fold[participant] = best_params_fold

        # Calculate mean score for inner folds
        print("Inner mean score:", np.mean(inner_scores), participant)

        # Get the best parameters from inner loop
        best_params = clf.best_params_
        print('Best parameters of', participant , '. participant:',  best_params)

        # Fit the selected model to the training set of outer CV
        # For prediction error estimation
        best_model = model.set_params(**best_params)
        best_model.fit(X_train_outer, y_train_outer)

        y_pred_test = best_model.predict(X_test_outer)

        # Store lists of true values and predictions 
        all_true_labels.extend(y_test_outer) #also for train set?
        all_predictions.extend(y_pred_test)

        # MSEs or accuracies from outer loop
        if task == 'regression':
            score_test.append(mean_squared_error(y_test_outer, y_pred_test))
            print("Mean Squared Error on Test Set:", score_test[-1], "in outer fold", participant)

        if task == 'classification':
            score_test.append(accuracy_score(y_test_outer, y_pred_test))
            print("Accuracy on Test Set:", score_test[-1], "in outer fold", participant)

    # Calculate the score across all folds in the outer loop
    mean_score = np.mean(score_test)
    print("Mean Mean Squared Error(regression) or accuracy(classification) in total: {:.2f}".format(mean_score))

    # Calculate the most common best parameter
    most_common_best_param = best_params_counts.most_common(1)[0][0]

    print("Most common best parameter:", most_common_best_param)

    # Output the first 10 elements of true labels and predictions
    print("True Labels (First 10 elements):", all_true_labels[:10])
    print("Predictions (First 10 elements):", all_predictions[:10])

    return mean_score, all_true_labels, all_predictions, score_test, most_common_best_param

# ____________________________________________________________________________
# Usage

# Directory
bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
data_path = opj(bidsroot)
# Load epochs object
epochs = mne.read_epochs(data_path, preload=True)

# Set target and label data
X = epochs.get_data()
# y = epochs.metadata["rating"].values  # .value correct? else, missing epochs are a problem
y = epochs.metadata["task"].values

# Define the groups (participants)
groups = epochs.metadata["participant_id"].values

#model= LogisticRegression()   
model = svm.SVC()
parameters = {"C": [1, 10, 100]}

#model = RandomForestClassifier()
#parameters = {"n_estimators": [1, 10, 100]}

#mean_score, all_true_labels, all_predictions, score_test = within_participant_DL(model, X, y, task='classification', nfolds=3, groups=groups)
mean_score, all_true_labels, all_predictions, score_test, most_common_best_param = within_participant_nested_cv(model, X, y, parameters = parameters, task = 'classification', nfolds=3, groups=groups)

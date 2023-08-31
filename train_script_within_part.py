#!/usr/bin/env python

import mne
import numpy as np
import torch
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, balanced_accuracy_score
from collections import Counter
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from skorch.dataset import Dataset
from tensorboardX import SummaryWriter
from sklearn.metrics import r2_score
from skorch.helper import predefined_split


def trainingDL_within(model, X, y, task='classification', nfolds=10, groups=None, writer=None):
    """
    Train and evaluate a machine learning model using cross-validation within participants.

    Parameters:
    - model (sklearn.base.BaseEstimator or torch.nn.Module): The machine learning model to train.
    - X (numpy.ndarray): Input data with shape (n_samples, n_features) or (n_samples, n_channels, n_time_points).
    - y (numpy.ndarray): Target labels or values.
    - task (str, optional): The task type, either 'regression' or 'classification'. Default is 'classification'.
    - groups (numpy.ndarray, optional): Group labels for grouping samples by participants. Default is None.

    Returns:
    - mean_score (float): Mean accuracy (for classification) or mean squared error (for regression) across participants.
    - all_true_labels (list): List of true class labels or values across all validation participants.
    - all_predictions (list): List of predicted class labels or values across all validation participants.
    """
    X = X.astype(np.float32)

    # Convert categorical labels to integer indices
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        y = torch.tensor(y, dtype=torch.int64)
    else:
        y = torch.tensor(y, dtype=torch.float32)

    # Initialize an array to store accuracy scores for each participant
    participant_scores = []
    # Initialize arrays to store true labels and predictions for each fold
    all_true_labels = []
    all_predictions = []

    # Get unique participant IDs
    unique_participants = np.unique(groups)
    train_iteration = 0



    # Loop over each participant
    for participant in unique_participants:
        fold_scores = []

        # Get the data indices for the current participant
        participant_indices = np.where(groups == participant)[0]

        X_part, y_part = X[participant_indices], y[participant_indices]

        # K fold cross validation
        if task ==  'classification':
            fold_split = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42).split(X_part, y_part)
        else:
            fold_split = KFold(n_splits=nfolds, shuffle=True, random_state=42).split(X_part, y_part)

        y_pred = np.empty_like(y_part)

        for train_idx, test_idx in fold_split:
            clf_fold = clone(model)

            # Split train and test
            X_train_fold, X_test = X_part[train_idx], X_part[test_idx]
            y_train_fold, y_test = y_part[train_idx], y_part[test_idx]

            # Split train and validation
            if task == 'regression':
                stratify = None
            else:
                stratify = y_train_fold

            X_train_fold, X_valid, y_train_fold, y_valid = train_test_split(
                X_train_fold,
                y_train_fold,
                test_size=0.2,
                stratify=stratify,
                shuffle=True,
                random_state=42,
            )

            if task == 'regression':
                y_valid = y_valid.unsqueeze(1)
                y_train_fold = y_train_fold.unsqueeze(1)

            valid_set = Dataset(X_valid, y_valid)
            clf_fold.set_params(
                **{"train_split": predefined_split(valid_set)}
            )

            clf_fold.fit(X_train_fold, y_train_fold)
            y_pred[test_idx] = clf_fold.predict(X_test).flatten()

            if task == 'regression':
                fold_scores.append(np.sqrt(mean_squared_error(y_test, y_pred[test_idx])))
                print("fold RMSE: ", fold_scores[-1])
            else:
                fold_scores.append(balanced_accuracy_score(y_test, y_pred[test_idx]))
                print("fold accuracy: ", fold_scores[-1])

        participant_scores.append(np.mean(fold_scores))

        # Append the true class names to the list
        all_true_labels.extend(y_part)
        # Append the predicted label strings to the list
        all_predictions.extend(y_pred)

    # Calculate the mean accuracy across all participants
    mean_score = np.mean(participant_scores)
    print("Mean Accuracy/RMSE across all participants: {:.3f}".format(mean_score))
    writer.close()

    return mean_score, all_true_labels[:], all_predictions[:], participant_scores

def training_nested_cv_within(model, X, y, parameters, task = 'regression', nfolds=4, n_inner_splits = 5, groups=None, writer=None):
    """
    Train and evaluate a machine learning model using nested cross-validation within participants.

    Parameters:
    - model (sklearn.base.BaseEstimator or torch.nn.Module): The machine learning model to train.
    - X (numpy.ndarray): Input data with shape (n_samples, n_features) or (n_samples, n_channels, n_time_points).
    - y (numpy.ndarray): Target labels or values.
    - parameters (dict): Dictionary of hyperparameters for grid search.
    - task (str, optional): The task type, either 'regression' or 'classification'. Default is 'regression'.
    - nfolds (int, optional): Number of folds for outer cross-validation. Default is 5.
    - groups (numpy.ndarray, optional): Group labels for grouping samples by participants. Default is None.

    Returns:
    - mean_score (float): Mean mean squared error (for regression) or accuracy (for classification) across participants and folds.
    - all_true_labels (list): List of true class labels or values across all validation participants and folds.
    - all_predictions (list): List of predicted class labels or values across all validation participants and folds.
    - score_test (list): List of mean squared errors (for regression) or accuracies (for classification) for each participant and fold.
    - most_common_best_param (str): Most common best parameter combination from grid search.
    """
    # Initialize arrays to store true labels and predictions for each fold
    all_true_labels = np.empty_like(y)
    all_predictions = np.empty_like(y)

    # Store scores from outer loop
    score_test = []

    # Initialize dictionaries to store best parameters and their occurrences
    best_params_counts = Counter()
    best_params_per_fold = {}

    # Create a pipeline for preprocessing
    if isinstance(model, Pipeline):
        full_pipe = model

    else:
        full_pipe = make_pipeline(
            mne.decoding.Scaler(scalings='mean'), # Scale the data
            mne.decoding.Vectorizer(), # Vectorize the data
            model # Add the ML model
        )

    # Outer cross-validation
    # Initialize GroupKFold with the desired number of folds
        # Get unique participant IDs
    unique_participants = np.unique(groups)
    participant_scores = []
    # Loop over each participant
    for participant in unique_participants:
        fold_scores = []
        print(participant)
        # Get the data indices for the current participant
        participant_indices = np.where(groups == participant)[0]

        # get plit participant data
        X_part, y_part = X[participant_indices], y[participant_indices]

        # K fold cross validation
        if task ==  'classification':
            fold_split = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42).split(X_part, y_part)
        else:
            fold_split = KFold(n_splits=nfolds, shuffle=True, random_state=42).split(X_part, y_part)


        y_pred = np.empty_like(y_part)

        # Outer cross-validation
        for train_idx, test_idx in fold_split:
            clf_fold = clone(full_pipe)

            # Split train and test
            X_train_fold, X_test = X_part[train_idx], X_part[test_idx]
            y_train_fold, y_test = y_part[train_idx], y_part[test_idx]

            # inner cross-validation
            # fit regressor to training data of inner CV
            clf_fold = GridSearchCV(clf_fold, parameters, cv = KFold(n_inner_splits).split(X_train_fold, y_train_fold), refit = True)
            clf_fold.fit(X_train_fold, y_train_fold)

            # # Store best parameters for each fold
            best_params_fold = clf_fold.best_params_
            best_params_counts.update([str(best_params_fold)])
            best_params_per_fold[participant] = best_params_fold


            y_pred[test_idx] = clf_fold.predict(X_test).flatten()


            if task == 'regression':
                fold_scores.append(np.sqrt(mean_squared_error(y_test, y_pred[test_idx])))
                print("fold RMSE: ", fold_scores[-1])
            else:
                fold_scores.append(balanced_accuracy_score(y_test, y_pred[test_idx]))
                print("fold accuracy: ", fold_scores[-1])

        participant_scores.append(np.mean(fold_scores))

        # Fit the selected model to the training set of outer CV
        # For prediction error estimation
        #best_model = model.set_params(**best_params)
        #y_pred_test = best_model.predict(X_test_outer)


        # Append the true class names to the list
        all_true_labels[participant_indices] = y_part #probably dont need
        # Append the predicted label strings to the list
        all_predictions[participant_indices] = y_pred 


        # MSEs or accuracies from outer loop
        # if task == 'regression':
        #     mse = mean_squared_error(y_test_outer, y_pred_test)
        #     score_test.append(mse)
        #     print("Mean Squared Error on Test Set:", score_test[-1], "in outer fold", participant)
        #     writer.add_scalar('Train Loss/MSE', mse, outer_train_iteration)

        # if task == 'classification':
        #     accuracy = accuracy_score(y_test_outer, y_pred_test)
        #     score_test.append(accuracy)
        #     print("Accuracy on Test Set:", score_test[-1], "in outer fold", participant)
        #     writer.add_scalar('Train Accuracy', accuracy, outer_train_iteration)

        # outer_train_iteration+=1

    # Calculate the score across all folds in the outer loop
    mean_score = np.mean(participant_scores)
    print("Mean Accuracy/RMSE across all participants: {:.3f}".format(mean_score))
    writer.close()
    # Calculate the most common best parameter
    most_common_best_param = best_params_counts.most_common(1)[0][0]

    print("Most common best parameter:", most_common_best_param)

    # Output the first 10 elements of true labels and predictions
    print("True Labels (First 10 elements):", all_true_labels[:10])
    print("Predictions (First 10 elements):", all_predictions[:10])
    writer.close()
    return mean_score, all_true_labels, all_predictions, score_test, most_common_best_param

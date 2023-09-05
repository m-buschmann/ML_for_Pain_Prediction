#!/usr/bin/env python

import mne
import numpy as np
import torch
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, GroupKFold, KFold, GridSearchCV, GroupShuffleSplit
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import Counter
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorboardX import SummaryWriter
from sklearn.metrics import r2_score
from skorch.dataset import Dataset
from skorch.helper import predefined_split



def group_train_valid_split(X, y, groups, proportion_valid=0.2):
    splitter = GroupShuffleSplit(
        test_size=proportion_valid, n_splits=2, random_state=42
    )
    split = splitter.split(X, groups=groups)
    train_inds, valid_inds = next(split)
    return Dataset(X[train_inds], y[train_inds]), Dataset(X[valid_inds], y[valid_inds])



def trainingDL_between(model, X, y, task = 'regression', nfolds=10, groups=None, writer=None):
    """
    Train and evaluate a machine learning model using cross-validation.

    Parameters:
    - model (sklearn.base.BaseEstimator or torch.nn.Module): The machine learning model to train.
    - X (numpy.ndarray): Input data with shape (n_samples, n_features) or (n_samples, n_channels, n_time_points).
    - y (numpy.ndarray): Target labels or values.
    - task (str, optional): The task type, either 'regression' or 'classification'. Default is 'regression'.
    - nfolds (int, optional): Number of folds for cross-validation. Default is 5.
    - groups (numpy.ndarray, optional): Group labels for grouping samples in cross-validation. Default is None.

    Returns:
    - mean_score (float): Mean mean squared error (for regression) or accuracy (for classification) across folds.
    - all_true_labels (list): List of true class labels or values across all validation folds.
    - all_predictions (list): List of predicted class labels or values across all validation folds.
    - score_test (float): Mean squared error (for regression) or accuracy (for classification) on the test set.
    """
    X = X.astype(np.int64)

    # Convert categorical labels to integer indices
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        y = torch.tensor(y, dtype=torch.int64)
    else:
        # Convert numerical labels to float
        y = torch.tensor(y, dtype=torch.float32)

    # Initialize GroupKFold with the desired number of folds
    gkf = GroupKFold(n_splits=nfolds)

    # Initialize arrays to store true labels and predictions for each fold
    all_true_labels = np.empty_like(y)
    all_predictions = np.empty_like(y)
    # Initialize an array to store accuracy/mse scores for each fold
    scores = []

    # Cross-validation loop
    for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

        # Copy untrained model (always safer to start fresh with each fold)
        model_fold = clone(model)

        # Split data into training and test sets based on the current fold indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # New groups from training data for split into train and val
        group2 = groups[train_index]

        all_true_labels[test_index] = y_test

        # Further split the training set into training and validation sets
        splitter = GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=42)
        split = splitter.split(X_train, groups=group2)
        train_inds, valid_inds = next(split)

        X_train, X_val = X_train[train_inds], X_train[valid_inds]
        y_train, y_val = y_train[train_inds], y_train[valid_inds]

        # Check it all makes sense
        assert (len(X_train) + len(X_val) + len(X_test)) == len(y)

        if task == "regression":
            # Unsqueeze to get same shape as output
            y_val = y_val.unsqueeze(1)
            y_train = y_train.unsqueeze(1)

        # Update the model to feed it the validation set during training
        model_fold.set_params(**{'train_split': predefined_split(Dataset(X_val, y_val))})

        # Fit the model on the training data
        model_fold.fit(X_train, y_train)
        # Predict on the test set
        y_pred = model_fold.predict(X_test).squeeze()

        all_predictions[test_index] = y_pred

        if task == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            scores.append(mse)
            print("Mean Squared Error in fold", i+1, ":", mse)
            writer.add_scalar('Train Loss/MSE', mse, i+1) 
            # Calculate R-squared score
            r2 = r2_score(y_test, y_pred)
            writer.add_scalar('Train R-squared', r2, i+1)

        if task == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            scores.append(accuracy)
            print("Accuracy in fold", i+1, ":", accuracy)
            writer.add_scalar('Train Accuracy', accuracy, i+1)

    # Calculate the mean mean squared error across all folds
    mean_score = np.mean(scores)
    print("Mean Mean Squared Error(regression) or accuracy(classification) over all folds: {:.2f}".format(mean_score))

    # Test the model on completely new data

    if task == 'regression':
        mse_test = mean_squared_error(all_true_labels, all_predictions)
        r2_test = r2_score(y_test, y_pred_test) #referenced before assignment

        print("Mean Squared Error total:", score_test)
        writer.add_scalar('Test Loss/MSE', mse_test)
        writer.add_scalar('Test R-squared', r2_test)

    if task == 'classification':
        score_test = accuracy_score(all_true_labels, all_predictions)
        print("Accuracy on Test Set:", score_test)
        # Convert y_test to integer type
        #y_test = y_test.astype(int)  # Ensure integer type
        # Convert the predicted integer indices to original class names
        y_test = label_encoder.inverse_transform(all_true_labels)
        y_pred_test = label_encoder.inverse_transform(all_predictions)
        writer.add_scalar('Test Accuracy', score_test)


    # Output the first 10 elements of true labels and predictions
    print("True Labels (First 10 elements):", all_true_labels[:10])
    print("Predictions (First 10 elements):", all_predictions[:10])
    # Close the SummaryWriter
    writer.close()
    return mean_score, all_true_labels, all_predictions, score_test


def training_nested_cv_between(model, X, y, parameters, task = 'regression', nfolds=4, n_inner_splits=5, groups=None, writer=None):
    """
    Perform nested cross-validation for training and evaluating a machine learning model.

    Parameters:
    - model (sklearn.base.BaseEstimator or torch.nn.Module): The machine learning model to train.
    - X (numpy.ndarray): Input data with shape (n_samples, n_features) or (n_samples, n_channels, n_time_points).
    - y (numpy.ndarray): Target labels or values.
    - parameters (dict): Hyperparameter grid for GridSearchCV.
    - task (str, optional): The task type, either 'regression' or 'classification'. Default is 'regression'.
    - nfolds (int, optional): Number of folds for outer cross-validation. Default is 5.
    - groups (numpy.ndarray, optional): Group labels for grouping samples in cross-validation. Default is None.

    Returns:
    - mean_score (float): Mean mean squared err/or (for regression) or accuracy (for classification) across outer folds.
    - all_true_labels (list): List of true class labels or values across all outer validation folds.
    - all_predictions (list): List of predicted class labels or values across all outer validation folds.
    - score_test (list): List of mean squared error (for regression) or accuracy (for classification) on the test set for each outer fold.
    - most_common_best_param (dict): Dictionary containing the most common best parameters from inner CV.
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
            #mne.decoding.Scaler(scalings='mean'), # data already normalized
            mne.decoding.Vectorizer(), # Vectorize the data
            model # Add the ML model
        )

    # Outer cross-validation
    # Initialize GroupKFold with the desired number of folds
    outer = GroupKFold(nfolds)
    for fold, (train_index_outer, test_index_outer) in enumerate(outer.split(X, y, groups)):
        X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
        y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

        inner_group = groups[train_index_outer]

        # inner cross-validation
        clf = GridSearchCV(full_pipe, parameters, cv=GroupKFold(n_inner_splits).split(X_train_outer, y_train_outer, inner_group),
                           refit=True)
        clf.fit(X_train_outer, y_train_outer) # Fit the model on the training data

        # Store best parameters for each fold
        best_params_fold = clf.best_params_
        #best_params_per_fold[fold] = best_params_fold #do we even need this?
        best_params_counts.update([str(best_params_fold)]) 
        
        y_pred_test = clf.predict(X_test_outer)

        # Store lists of true values and predictions 
        all_true_labels[test_index_outer] = y_test_outer#probably dont nee
        all_predictions[test_index_outer] = y_pred_test #added again?

        # MSEs or accuracies from outer loop
        if task == 'regression':
            score_test.append(mean_squared_error(y_test_outer, y_pred_test))
            print("Mean Squared Error on Test Set:", score_test[fold], "in outer fold", fold+1)
            writer.add_scalar('Test Loss/MSE', score_test[fold], fold+1) 

        if task == 'classification':
            score_test.append(accuracy_score(y_test_outer, y_pred_test))
            print("Accuracy on Test Set:", score_test[fold], "in outer fold", fold+1)
            writer.add_scalar('Test Accuracy', score_test[fold], fold+1) 

    # Calculate the score across all folds in the outer loop
    mean_score = np.mean(score_test)
    print("Mean Mean Squared Error(regression) or accuracy(classification) in total: {:.2f}".format(mean_score))

    # Calculate the most common best parameter
    most_common_best_param = best_params_counts.most_common(1)[0][0]
    print("Most common best parameter:", most_common_best_param)

    # Output the first 10 elements of true labels and predictions
    print("True Labels (First 10 elements):", all_true_labels[:10])
    print("Predictions (First 10 elements):", all_predictions[:10])
    writer.close()
    return mean_score, all_true_labels, all_predictions, score_test, most_common_best_param

    
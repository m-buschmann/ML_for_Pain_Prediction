# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2023-08-25 13:46:44
# @Last Modified by:   Your name
# @Last Modified time: 2023-08-25 14:06:44
#!/usr/bin/env python

import mne
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GroupKFold, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorboardX import SummaryWriter
from sklearn.metrics import r2_score



def trainingDL_between(model, X, y, task = 'regression', nfolds=4, n_inner_splits=5, groups=None, writer=None):
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
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = torch.tensor(y, dtype=torch.int64)

    # Initialize GroupKFold with the desired number of folds
    gkf = GroupKFold(n_splits=nfolds)
    
    # Initialize arrays to store true labels and predictions for each fold
    all_true_labels = []
    all_predictions = []
    # Initialize an array to store accuracy/mse scores for each fold
    scores = []
    # Create a pipeline for preprocessing and classification
    pipe = make_pipeline(
        mne.decoding.Scaler(scalings='mean'), # Scale the data
        model
    )

    # Cross-validation loop
    for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):

        # Split data into training and test sets based on the current fold indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # New groups from training data for split into train and val
        group2 = groups[train_index]
        # Further split the training set into training and validation sets
        gkf2 = GroupKFold(n_splits=n_inner_splits)
        # Further split the training set into training and validation sets
        for train_idx, val_idx in gkf2.split(X_train, y_train, groups=group2):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        # Shuffle within the training set
        train_shuffle_indices = np.random.permutation(len(y_train))
        X_train, y_train = X_train[train_shuffle_indices], y_train[train_shuffle_indices]
        
        # Fit the model on the training data
        pipe.fit(X_train, y_train)
        # Predict on the test set
        y_pred = pipe.predict(X_val)
        
        if task == 'regression':
            mse = mean_squared_error(y_val, y_pred)
            scores.append(mse)
            print("Mean Squared Error in fold", i+1, ":", mse)
            writer.add_scalar('Train Loss/MSE', mse, i+1) 
            # Calculate R-squared score
            r2 = r2_score(y_val, y_pred)
            writer.add_scalar('Train R-squared', r2, i+1)

        if task == 'classification':
            accuracy = accuracy_score(y_val, y_pred)
            scores.append(accuracy)
            print("Accuracy in fold", i+1, ":", accuracy)
            writer.add_scalar('Train Accuracy', accuracy, i+1)

    # Calculate the mean mean squared error across all folds
    mean_score = np.mean(scores)
    print("Mean Mean Squared Error(regression) or accuracy(classification) over all folds: {:.2f}".format(mean_score))

    # Test the model on completely new data
    score_test = []
    y_pred_test = pipe.predict(X_test)

    if task == 'regression':
        score_test = mean_squared_error(y_test, y_pred_test)
        print("Mean Squared Error on Test Set:", score_test)
        # Convert the list of tensors to a numpy array of floats
        y_test = np.array([tensor.item() for tensor in y_test])
        # Concatenate the NumPy arrays in the predictions list
        y_pred_test = [prediction[0].item() for prediction in y_pred_test]
        writer.add_scalar('Test Loss/MSE', score_test)
        r2 = r2_score(y_test, y_pred_test)
        writer.add_scalar('Test R-squared', r2)

    if task == 'classification':
        score_test = accuracy_score(y_test, y_pred_test)
        print("Accuracy on Test Set:", score_test)
        # Convert y_test to integer type
        #y_test = y_test.astype(int)  # Ensure integer type
        # Convert the predicted integer indices to original class names
        y_test = label_encoder.inverse_transform(y_test)
        y_pred_test = label_encoder.inverse_transform(y_pred_test)
        writer.add_scalar('Test Accuracy', score_test)

    # Append the true class names to the lis
    all_true_labels.extend(y_test)
    # Append the predicted label strings to the list
    all_predictions.extend(y_pred_test)
    
    # Output the first 10 elements of true labels and predictions
    print("True Labels (First 10 elements):", all_true_labels[:10])
    print("Predictions (First 10 elements):", all_predictions[:10])
    # Close the SummaryWriter
    writer.close()
    return mean_score, all_true_labels, all_predictions, score_test


def training_nested_cv_between(model, X, y, parameters, task = 'regression', nfolds=4, n_inner_splits = 5, groups=None, writer=None):
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
    - mean_score (float): Mean mean squared error (for regression) or accuracy (for classification) across outer folds.
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
    preprocessing_pipe = make_pipeline(
        mne.decoding.Scaler(scalings='mean'), # Scale the data
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
        clf = GridSearchCV(model, parameters, cv=GroupKFold(n_inner_splits).splits(X_train_outer, y_train_outer, inner_group),
                           refit=True)
        clf.fit(X_train_outer, y_train_outer) # Fit the model on the training data

            # Store best parameters for each fold
        best_params_fold = clf.best_params_
        best_params_per_fold[fold] = best_params_fold

        inner_train_iteration+=1
        
        y_pred_test = clf.predict(X_test_outer)

        # Store lists of true values and predictions 
        all_true_labels[test_index_outer] = y_test_outer

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

    
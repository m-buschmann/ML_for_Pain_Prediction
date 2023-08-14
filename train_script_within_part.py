import mne
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def trainingDL_within(model, X, y, task='classification', groups=None):
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
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = torch.tensor(y, dtype=torch.int64)

    # Initialize an array to store accuracy scores for each participant
    participant_scores = []
    # Initialize arrays to store true labels and predictions for each fold
    all_true_labels = []
    all_predictions = []
    # Create a pipeline for preprocessing and classification
    pipe = make_pipeline(
        mne.decoding.Scaler(scalings='mean'),  # Scale the data
        model
    )

    # Get unique participant IDs
    unique_participants = np.unique(groups)

    # Loop over each participant
    for participant in unique_participants:

        # Get the data indices for the current participant
        participant_indices = np.where(groups == participant)[0]
        # Split participant data into training and testing using train_test_split
        train_idx, test_idx = train_test_split(participant_indices, test_size=0.2, random_state=42)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit the model on the training data
        pipe.fit(X_train, y_train)
        # Predict on the test set
        y_pred = pipe.predict(X_test)
            
        if task == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            participant_scores.append(mse)
            print("Participant", participant, "MSE:", mse)
            # Convert the list of tensors to a numpy array of floats
            y_test = np.array([tensor.item() for tensor in y_test])
            # Concatenate the NumPy arrays in the predictions list
            y_pred = [prediction[0].item() for prediction in y_pred]

        if task == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            participant_scores.append(accuracy)
            print("Participant", participant, "Accuracy:", accuracy)
            # Convert the predicted integer indices to original class names
            y_test = label_encoder.inverse_transform(y_test)
            y_pred = label_encoder.inverse_transform(y_pred)
    
    # Append the true class names to the list
    all_true_labels.extend(y_test)
    # Append the predicted label strings to the list
    all_predictions.extend(y_pred)

    # Output the first 10 elements of true labels and predictions
    print("True Labels (First 10 elements):", all_true_labels[:10])
    print("Predictions (First 10 elements):", all_predictions[:10])

    # Calculate the mean accuracy across all participants
    mean_score = np.mean(participant_scores)
    print("Mean Accuracy/MSE across all participants: {:.3f}".format(mean_score))

    return mean_score, all_true_labels, all_predictions

def training_nested_cv_within(model, X, y, parameters, task = 'regression', nfolds=5, groups=None):
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

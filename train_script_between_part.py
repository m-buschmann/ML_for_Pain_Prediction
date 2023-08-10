import mne
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GroupKFold, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def trainingDL_between(model, X, y, task = 'regression', nfolds=5, groups=None):
    X = X.astype(np.float32)

    
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
        mne.decoding.Vectorizer(), # Vectorize the data
        print(X.shape, "(n_epochs, n_channels, n_time_points)"),
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
        gkf2 = GroupKFold(n_splits=2)
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

        # Convert the predicted integer indices to original class names
        y_true_class_names = label_encoder.inverse_transform(y_val)
        # Append the true class names to the list
        all_true_labels.extend(y_true_class_names)
        
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        # Append the predicted label strings to the list
        all_predictions.extend(y_pred_labels)

        # Append true labels and predictions to the corresponding lists
        """all_true_labels.extend(y_val) #also for test set?
        all_predictions.extend(y_pred)"""
        
        if task == 'regression':
            mse = mean_squared_error(y_val, y_pred)
            scores.append(mse)
            print("Mean Squared Error in fold", i+1, ":", mse)

        if task == 'classification':
            accuracy = accuracy_score(y_val, y_pred)
            scores.append(accuracy)
            print("Accuracy in fold", i+1, ":", accuracy)

    # Calculate the mean mean squared error across all folds
    mean_score = np.mean(scores)
    print("Mean Mean Squared Error(regression) or accuracy(classification) over all folds: {:.2f}".format(mean_score))


    # Output the first 10 elements of true labels and predictions
    print("True Labels (First 10 elements):", all_true_labels[:10])
    print("Predictions (First 10 elements):", all_predictions[:10])

    # Test the model on completely new data
    score_test = []
    y_pred_test = pipe.predict(X_test)

    if task == 'regression':
        score_test = mean_squared_error(y_test, y_pred_test)
        print("Mean Squared Error on Test Set:", score_test)

    if task == 'classification':
        score_test = accuracy_score(y_test, y_pred_test)
        print("Accuracy on Test Set:", score_test)

    return mean_score, all_true_labels, all_predictions, score_test


def training_nested_cv_between(model, X, y, parameters, task = 'regression', nfolds=5, groups=None):
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
    outer = GroupKFold(nfolds)
    for fold, (train_index_outer, test_index_outer) in enumerate(outer.split(X, y, groups)):
        X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
        y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

        inner_group = groups[train_index_outer]

        inner_scores = []
        # inner cross-validation
        inner = GroupKFold(2) #increase with more data
        for train_index_inner, test_index_inner in inner.split(X_train_outer, y_train_outer, inner_group):
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
            best_params_per_fold[fold] = best_params_fold

        # Calculate mean score for inner folds
        print("Inner mean score:", np.mean(inner_scores), fold+1 , '.fold',)

        # Get the best parameters from inner loop
        best_params = clf.best_params_
        print('Best parameters of', fold+1 , '.fold:',  best_params)

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
            print("Mean Squared Error on Test Set:", score_test[fold], "in outer fold", fold+1)

        if task == 'classification':
            score_test.append(accuracy_score(y_test_outer, y_pred_test))
            print("Accuracy on Test Set:", score_test[fold], "in outer fold", fold+1)

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

    
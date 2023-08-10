from sklearn.model_selection import StratifiedGroupKFold
import mne
from os.path import join as opj
import numpy as np
from sklearn import datasets, linear_model, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GroupKFold, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import operator

# set random state
state = 1

# Directory
bidsroot = '/home/mathilda/MITACS/Project/eeg_pain_v2/derivatives/cleaned epochs/single_sub_cleaned_epochs/sub_3_to_5_cleaned_epo.fif'
data_path = opj(bidsroot)
# Load epochs oject
epochs = mne.read_epochs(data_path, preload=True)

# Create a pipeline for preprocessing
preprocessing_pipe = make_pipeline(
    mne.decoding.Scaler(scalings='mean'), # Scale the data
    mne.decoding.Vectorizer() # Vectorize the data
)

# Set target and label data
X = epochs.get_data()
#y = epochs.metadata["rating"].values # .value correct? else, missing epochs are a problem
y = epochs.metadata["task"].values

X = preprocessing_pipe.fit_transform(X)

# Get groups from the metadata DataFrame
groups = epochs.metadata["participant_id"].values

# Set up possible values of parameters to optimize over
hyperparameter_space = {"C": [1, 10, 100]}

# Create a pipeline for classification with LogisticRegression
model = LogisticRegression() # Logistic Regression Classifier

outer_scores = []
inner_group = [] 

# outer cross-validation
# Initialize StratifiedGroupKFold with the desired number of folds
outer = StratifiedGroupKFold(n_splits=3)
for fold, (train_index_outer, test_index_outer) in enumerate(outer.split(X, y, groups)):
    X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
    y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

    inner_mean_scores = []
    inner_group = groups[train_index_outer]

    # define explored parameter space.
    # procedure below should be equal to GridSearchCV
    tuned_parameter = [1, 10, 100]
    for param in tuned_parameter:

        inner_scores = []
        # inner cross-validation
        inner = StratifiedGroupKFold(2)
        for train_index_inner, test_index_inner in inner.split(X_train_outer, y_train_outer, inner_group):
            # split the training data of outer CV
            X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
            y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

            # fit regressor to training data of inner CV
            clf = LogisticRegression(C=param, n_jobs=-1, random_state=1)
            clf.fit(X_train_inner, y_train_inner)
            inner_scores.append(clf.score(X_test_inner, y_test_inner))

        # calculate mean score for inner folds
        inner_mean_scores.append(np.mean(inner_scores))

    # get maximum score index
    index, value = max(enumerate(inner_mean_scores), key=operator.itemgetter(1))

    print('Best parameter of %i fold: %i' % (fold + 1, tuned_parameter[index]))

    # fit the selected model to the training set of outer CV
    # for prediction error estimation
    clf2 = LogisticRegression(C=tuned_parameter[index], n_jobs=-1, random_state=1)
    clf2.fit(X_train_outer, y_train_outer)
    outer_scores.append(clf2.score(X_test_outer, y_test_outer))

# show the prediction error estimate produced by nested CV
print ('Unbiased prediction score: %.4f' % (np.mean(outer_scores)))
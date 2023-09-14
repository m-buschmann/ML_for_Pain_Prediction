# ML_for_Pain_Prediction

## overview
code: 
- 01eeg_preprocess.py: preprocessing of the EEG data
- 02cleaning_epochs.py: rejecting epochs, adding metadata, resample, build one file out of all the epochs (still contains some code in the end that's only necessary with low memory, we can remove that when it's no longer needed)
- train_script_between_part.py: contains training for DL and standard ML between participants
- train_script_within_part.py: contains training for DL and standard ML within participants
- bayes_train_script_between_part.py: contains training for DL and standard ML between participants with bayes nested CV
- bayes_train_script_within_part.py: contains training for DL and standard ML within participants with bayes nested CV
- 04bayes_models_cc.py: Run the training with bayes grid search, needed it right now as its own file, but can be integrated into the other models script as an optiomn later
- 04models.py basically the same as "04models_cc.py", used it to experiment in order to not mess anything up in the code I run in Compute Canada. Can be deleted.
- 04models_cc.py: The code to run on Compute Canada to train models. Should always contain the current state of the code. To run it on Compute Canada, it expects the following arguments:
    model_name = sys.argv[1]\
    part = sys.argv[2]\
    target = sys.argv[3]\
  (as from 14.09.2023)
- 05training_whole_data.py: Train the model on the whole datase, save the model. Expects the arguments
    model_name = sys.argv[1]\
    part = sys.argv[2]\
    target = sys.argv[3]\
    search_params = sys.argv[4] (True or False, whether to do Gridsearch to find Hyperparameters. If False, best Parameters need to inserted in the code)
- 06_test_model.py: Test the model on new data. Still needs to be changed to use Aly's data (as of 14.09.2023)
- tensorboard.py: show the tensorboard logs
- preprocess_X.py: Remove epochs with too high difference in intensity and standardize X. This data is only used on Compute Canada. On other computers, this will be done when running 04models_cc.py

results:
- folder "results": the results from Compute Canada. Needs to be updated when we have all results
    - confusion matrices: contains the confusion matrices from the classification task for all models
    - csvs: contains the files with scores, true labels, predicted labels and for standard ML the most common best parameters
    - logs: contains the tensorboard logs (accuracies, RMSE, r2). If there are several logs for one model, the last one is the latest. Inspect the logs either by running "tensorboard --logdir /path/to/log --port 6007" in a terminal or with the code tensorboard.py
- folder "models": contains the models after training on whole data
- folder "plots intensity and rating": contains the plots of comparision rating and intensity
- plots xxx: results computed from my laptop, delete later
- resultsxxx: results computed from my laptop, delete later
- images: results computed from my laptop, delete later
- logs: results computed from my laptop, delete later

## How to use
### Preprocess data
- 01eeg_preprocess.py: exchange bidsroot for the path to your eeg data
- 02cleaning_epochs.py: exchange bidsroot for the path to your eeg data
- preprocess_X.py: exchange bidsroot and log_dir for the path to your eeg data

### Run the training
- 04models_cc.py
    - if running on Compute Canada: easiest to use a .sh file with the arguments
    model_name = sys.argv[1] (the model to use)\
    part = sys.argv[2] ('within' or 'between')\
    target = sys.argv[3] (3_classes, 5_classes, intensity, or rating)\
    (see example below)
    - else: set your own bidsroot and log_dir and the arguments
    - right now, the code sets target = 'intensity' or '3_classes' later in the code to prevent mix-ups

### Train models on whole dataset
- 05training_whole_data.py
    - if running on Compute Canada: easiest to use a .sh file with the arguments
    model_name = sys.argv[1] (the model to use)\
    part = sys.argv[2] ('within' or 'between')\
    target = sys.argv[3] (3_classes, 5_classes, intensity, or rating)\
    search_params = sys.argv[4] (whether to do grid search or not)
    - else: set your own bidsroot, log_dir, model_dir (where to save models) and the arguments

### Test on new data
- 06_test_model.py
    - if running on Compute Canada: easiest to use a .sh file with the arguments
    model_name = sys.argv[1] (the model to use)\
    part = sys.argv[2] ('within' or 'between')\
    target = sys.argv[3] (3_classes, 5_classes, intensity, or rating)
    - else: set your own bidsroot,  log_dir, model_dir (where to load models from) and the arguments

        
## example .sh file:   
#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=def-mpcoll
#SBATCH --gpus=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G
#SBATCH --mail-user=mathilda.buschmann.1@ulaval.ca
#SBATCH --mail-type=ALL
#SBATCH --chdir=/lustre04/scratch/mabus103

module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/env 

source $SLURM_TMPDIR/env/bin/activate

cp --verbose eegdecode_venv_wheel.zip $SLURM_TMPDIR/eegdecode_venv_wheel.zip

unzip -o -n $SLURM_TMPDIR/eegdecode_venv_wheel.zip -d $SLURM_TMPDIR

pip install --no-index --find-links=$SLURM_TMPDIR/eegdecode_venv_wheel -r $SLURM_TMPDIR/eegdecode_venv_wheel/requirements.txt

pip list

#model type, within or between particiants, target
python /lustre04/scratch/mabus103/ML_for_Pain_Prediction/04models_cc.py "deep4netRegression" "within" 'intensity'


#models:
#"LogisticRegression"
#"LinearRegression"

#"SVC"
#"SVR"

#"RFClassifier"
#"RFRegressor"

#"ElasticNet"
#"SGD"

#"deep4netClassification"
#"deep4netRegression"

#"shallowFBCSPNetClassification"
#"shallowFBCSPNetRegression"


#targets
#'3_classes'
#'5_classes'
#'rating'
#'intensity'


    

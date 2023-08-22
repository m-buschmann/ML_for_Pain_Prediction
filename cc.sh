#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --account=def-mpcoll
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=186G
#SBATCH --chdir=/lustre04/scratch/mabus103

module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/env 

source $SLURM_TMPDIR/env/bin/activate

cp --verbose eegdecode_venv_wheel.zip $SLURM_TMPDIR/eegdecode_venv_wheel.zip

unzip -o -n $SLURM_TMPDIR/eegdecode_venv_wheel.zip -d $SLURM_TMPDIR

pip install --no-index --find-links=$SLURM_TMPDIR/eegdecode_venv_wheel -r $SLURM_TMPDIR/eegdecode_venv_wheel/requirements.txt

pip list

python /lustre04/scratch/mabus103/ML_for_Pain_Prediction/04models_cc.py

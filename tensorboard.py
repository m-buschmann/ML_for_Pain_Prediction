# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2023-08-25 13:46:44
# @Last Modified by:   Your name
# @Last Modified time: 2023-08-25 14:14:42
#!/usr/bin/env python

from os.path import join as opj
from sklearn.pipeline import make_pipeline
from tensorboardX import SummaryWriter

#tensorboard --logdir /home/mathilda/MITACS/Project/code/ML_for_Pain_Prediction/logs/deep4netClassification/between --port 6007

import subprocess

logdir = "/home/mathilda/MITACS/Project/CC/results/logs/SGD/between"
port = 6008

# Construct the TensorBoard command
command = f"tensorboard --logdir {logdir} --port {port}"

# Run TensorBoard using subprocess
process = subprocess.Popen(command, shell=True)

# Wait for TensorBoard to finish (you can remove this line if you want it to run in the background)
#process.wait()

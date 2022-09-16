# interpretableAI_DRP
## Overview
The code for running each model is divided into individual sub-folders. Two types of model execution can be done: 1) run a pretrained model with specified hyperparameters; 2) run model from scratch using specified hyperparameters. The former execution can be done by running the *run_pretrained.sh* script and the latter can be done by running *run_model_with_hype.sh* script.

Hyperparameter tuning has been performed on the validation set and the set set of hyperparameters for each validation strategy (leave-ccls-out, leave-drugs-out, leave-pairs-out) and each pathway collection (KEGG, PID, Reactome) are provided in sub-folders named *best_hyp*. 

## Input Data
The input data for the models can be found at (link).

## Environment Requirement
- python -3.9.7
- pytorch -1.11.0
- pandas -1.3.4
- numpy -1.20.3
- scipy -1.7.1
- scikit-learn -0.24.2

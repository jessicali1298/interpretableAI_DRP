# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:42:58 2022

@author: jessi
"""

import torch 
import os, sys, argparse, json
# os.environ['NUMEXPR_MAX_THREADS']='6'
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
print("the number of cpu threads: {}".format(torch.get_num_threads()))
torch.set_num_threads(6)
print("the number of cpu threads: {}".format(torch.get_num_threads()))

parent_dir = dirname(dirname(abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir

print(os.environ["PYTHONPATH"])

import pandas as pd
import numpy as np
from MLP.validation import prep_data, train_val, hyper_tune_main

from utils.utils import mkdir
from utils.eval_result import export_results, eval_metrics, eval_metrics_by
#%% parse_parameters
def parse_parameters():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataroot",
                        required=True,
                        help = "path for input data")
    parser.add_argument("--outroot",
                        required=True,
                        help = "path to save results")
    parser.add_argument("--hyproot",
                        required=False,
                        help = "path for hyperparameter dictionary, not required if tuning is True")
    parser.add_argument("--pathway",
                        required=True,
                        help = "name of pathway collection (KEGG, PID, Reactome)")
    parser.add_argument("--foldtype",
                        required=True,
                        help = "type of validation scheme (pair, drug, cl)")
    parser.add_argument("--drug_feature_type",
                        required=True,
                        help = "specify drug feature type (Morgan fingerprints (fp) or drug target (target))")
    parser.add_argument("--run_pretrained",
                        required=False,
                        action="store_true",
                        help = "whether to run pretrained model or not")
    parser.add_argument("--modelroot",
                        required=False,
                        help = "path for pretrained model to be run, required if run_pretrained is True")
    parser.add_argument("--tuning",
                        required=False,
                        action="store_true",
                        help = "whether to perform hyperparameter tuning or not")
    parser.add_argument("--gridroot",
                        required=False,
                        help = "path for hyperparameter tuning grid, required if tuning is True")
    parser.add_argument("--max_tuning_epoch",
                        required=False,
                        type=int,
                        help = "max number of training epoch to run for hyperparameter tuning, required if tuning is True")
    parser.add_argument("--num_samples",
                        required=False,
                        type=int,
                        help = "number of hyperparameter sets to be tested, required if tuning is true")
    return parser.parse_args()


#%% main
if __name__ == '__main__':
    args = parse_parameters()
    mkdir(args.outroot)
    sys.stdout = open(args.outroot  + args.foldtype + '_' + args.pathway + '_' + 'log.txt', 'w')
 
    # import data
    input_data_path = args.dataroot + 'model_agnostic_data/' + args.pathway + '/'
    indices = pd.read_csv(input_data_path + 'cl_drug_indices.csv', header=0)
    cl_features = pd.read_csv(input_data_path + 'rnaseq_fpkm.csv', header=0, index_col=0).to_numpy()
    
    if args.drug_feature_type=='fp':
        drug_features = pd.read_csv(input_data_path + 'drug_fp_matrix.csv', header=0, index_col=0).to_numpy()
    elif args.drug_feature_type=='target':
        drug_features = pd.read_csv(input_data_path + 'drug_target_matrix.csv', header=0, index_col=0).to_numpy()
    label_matrix = pd.read_csv(input_data_path + 'ic50_matrix.csv', header=0, index_col=0).to_numpy()
    foldtype = args.foldtype + '_fold'
    
    #%% run_pretrained
    # ------------------------------------------- run pretrained model ------------------------------------
    if args.run_pretrained==True:
        # load hyperparameter dictionary
        with open(args.hyproot) as f:
            data = f.read()
        hyp = json.loads(data)
        
        # partition dataset into train and test
        trainset, testset = prep_data(cl_features, drug_features, label_matrix, indices, 
                                          fold_type=foldtype, train_fold=[0,1,2], val_fold=[4])  
        
        # load pretrained model
        model_path = args.modelroot
        
        y, pred, metric_matrix = train_val(hyp, trainset, testset, 
                                           fold_type=foldtype, 
                                           load_pretrain=True, 
                                           model_path=model_path)
        
        # export predictions
        result_path = args.outroot + 'result_metrics' # model result path
        mkdir(result_path)
        result = export_results(indices, result_path + '/' + args.pathway + '_' + foldtype + '_result.csv', 
                                foldtype, y, pred, val_fold=[4])
    
    # ------------------------- running model from scratch (not loading pretrained ---------------------------
    else:
        #%% run with tuning
        # ---------- perform hyperparameter tuning and run model with optimal hyperparameters ----------------
        if args.tuning==True:
            # load hyperparameter tuning grid
            with open(args.gridroot) as f:
                grid = f.read()
            config = json.loads(grid)

            # partition dataset into train and validation set
            trainset, valset = prep_data(cl_features, drug_features, label_matrix, indices, 
                                         fold_type=foldtype, train_fold=[0,1,2], val_fold=[3])
            
            epoch, epoch_avg5, best_trial_config = hyper_tune_main(trainset, valset, 
                                                                   fold_type=foldtype, 
                                                                   grid=config,
                                                                   num_train_epoch=args.max_tuning_epoch, 
                                                                   num_samples=args.num_samples)
            
            # export most optimal set of hyperparameters
            best_trial_config["epoch"] = epoch_avg5
            x = json.dumps(best_trial_config)
            f = open(args.outroot + "/best_model_hyp_" + foldtype + '_' + args.pathway + ".txt","w")
            f.write(x)
            f.close()
            
            # run model with most optimal hyperparameters
            param_save_path = args.outroot + 'model_weights'
            hyp_save_path = args.outroot + 'model_hyp'
            metric_save_path = args.outroot + 'model_train_metrics'
            
            mkdir(param_save_path)
            mkdir(hyp_save_path)
            mkdir(metric_save_path)
            
            # partition dataset into train and testset
            trainset, testset = prep_data(cl_features, drug_features, label_matrix, indices, 
                                          fold_type=foldtype, train_fold=[0,1,2], val_fold=[4])   
    

            y, pred, metric_matrix = train_val(best_trial_config, trainset, testset, 
                                               fold_type=foldtype,  
                                               load_pretrain=False, model_path=None, 
                                               param_save_path=param_save_path, 
                                               hyp_save_path=hyp_save_path, 
                                               metric_save_path=metric_save_path, 
                                               description=foldtype + '_' + args.pathway)
            
          
        #%% run with specified hyp   
        # ---------------- run model without tuning and using specified hyperparameter ---------------------
        else:
            # load hyperparameter dictionary
            with open(args.hyproot) as f:
                data = f.read()
            hyp = json.loads(data)
            
            # partition dataset into train and testset
            trainset, testset = prep_data(cl_features, drug_features, label_matrix, indices, 
                                          fold_type=foldtype, train_fold=[0,1,2], val_fold=[4])   
    
            # run model with most optimal hyperparameters
            param_save_path = args.outroot + 'model_weights'
            hyp_save_path = args.outroot + 'model_hyp'
            metric_save_path = args.outroot + 'model_train_metrics'
            
            mkdir(param_save_path)
            mkdir(hyp_save_path)
            mkdir(metric_save_path)

            y, pred, metric_matrix = train_val(hyp, trainset, testset, 
                                               fold_type=foldtype,  
                                               load_pretrain=False, model_path=None, 
                                               param_save_path=param_save_path, 
                                               hyp_save_path=hyp_save_path, 
                                               metric_save_path=metric_save_path, 
                                               description=foldtype + '_' + args.pathway)


        # export predictions
        result_path = args.outroot + 'result_metrics' # model result path
        mkdir(result_path)
        result = export_results(indices, result_path + '/' + args.pathway + '_' + foldtype + '_result.csv', 
                                foldtype, y, pred, val_fold=[4])






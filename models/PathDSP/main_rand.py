# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 11:58:43 2022

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
from PathDSP.validation import prep_data, train_val, hyper_tune_main

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
    parser.add_argument("--run_pretrained",
                        required=False,
                        action="store_true",
                        help = "whether to run pretrained model or not")
    parser.add_argument("--modelroot",
                    required=False,
                    help = "path for pretrained model to be run, required if run_pretrained is True")
    return parser.parse_args()


#%% main
if __name__ == '__main__':
    args = parse_parameters()
    mkdir(args.outroot)
    sys.stdout = open(args.outroot  + args.foldtype + '_' + args.pathway + '_rand' + '_log.txt', 'w')
    
    
    # load hyperparameter dictionary
    with open(args.hyproot) as f:
        data = f.read()
    hyp = json.loads(data)
    
    common_data_path = args.dataroot + 'model_agnostic_data/' + args.pathway + '/'
    
    indices = pd.read_csv(common_data_path + 'cl_drug_indices.csv', header=0)
    drug_fp = pd.read_csv(common_data_path + 'drug_fp_matrix.csv', header=0, index_col=0).to_numpy()
    label_matrix = pd.read_csv(common_data_path + 'ic50_matrix.csv', header=0, index_col=0).to_numpy()
    foldtype = args.foldtype + '_fold'
    
    for i in range(3):
        rand_input_path = args.dataroot + 'PathDSP_rand/' + args.pathway + '/rand_matrix_' + str(i) + '/'
        cl_exp = pd.read_csv(rand_input_path + 'exp_NES.csv', header=0, index_col=0).to_numpy()
        cl_cnv = pd.read_csv(rand_input_path + 'cnv_zscore.csv', header=0, index_col=0).to_numpy()
        cl_mut = pd.read_csv(rand_input_path + 'mut_zscore.csv', header=0, index_col=0).to_numpy()
        drug_target = pd.read_csv(rand_input_path + 'target_zscore.csv', header=0, index_col=0).to_numpy()
        
        # run model with most optimal hyperparameters
        param_save_path = args.outroot + 'matrix_' + str(i) + '_model_weights'
        hyp_save_path = args.outroot + 'matrix_' + str(i) + '_model_hyp'
        metric_save_path = args.outroot + 'matrix_' + str(i) + '_model_train_metrics'
        
        mkdir(param_save_path)
        mkdir(hyp_save_path)
        mkdir(metric_save_path)
        
        # partition dataset into train and testset
        trainset, testset = prep_data(cl_exp, cl_mut, cl_cnv, drug_fp, drug_target,
                                     label_matrix, indices, fold_type=foldtype, 
                                     train_fold=[0,1,2], val_fold=[4]) 
    
        y, pred, metric_matrix = train_val(hyp, trainset, testset, 
                                          fold_type=foldtype,  
                                          load_pretrain=False, model_path=None, 
                                          param_save_path=param_save_path, 
                                          hyp_save_path=hyp_save_path, 
                                          metric_save_path=metric_save_path, 
                                          description=foldtype + '_' + args.pathway)
      
        # export predictions
        result_path = args.outroot + 'matrix_' + str(i) + '_result_metrics' # model result path
        mkdir(result_path)
        result = export_results(indices, result_path + '/' + args.pathway + '_' + foldtype + '_result.csv', 
                                foldtype, y, pred, val_fold=[4])

 
  
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
from HiDRA.validation import prep_data, train_val, hyper_tune_main

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
                        help = "specify drug feature type (Morgan fingerprints or drug target)")
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
    
    # import data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_data_path = args.dataroot + 'HiDRA/' + args.pathway + '/'
    common_data_path = args.dataroot + 'model_agnostic_data/' + args.pathway + '/'
    
    # model-specific data
    n_member_genes_path = input_data_path + 'n_member_genes.csv'
    n_pathway_members = torch.Tensor(pd.read_csv(n_member_genes_path, index_col=0).to_numpy())
    
    # model-agnostic data
    indices = pd.read_csv(common_data_path + 'cl_drug_indices.csv', header=0)
    cl_features = pd.read_csv(common_data_path + 'rnaseq_fpkm.csv', header=0, index_col=0).to_numpy()
    if args.drug_feature_type == 'fp':
        drug_features = pd.read_csv(common_data_path + 'drug_fp_matrix.csv', header=0, index_col=0).to_numpy()
        
    elif args.drug_feature_type == 'target':
        drug_features = pd.read_csv(common_data_path + 'drug_target_matrix.csv', header=0, index_col=0).to_numpy()    
    label_matrix = pd.read_csv(common_data_path + 'ic50_matrix.csv', header=0, index_col=0).to_numpy()
    
    foldtype = args.foldtype + '_fold'
    
    for i in range(20):
        pathway_data_path = input_data_path + 'random_pathway_matrices/' + 'matrix_' + str(i) + '.npy'
        
        pathway_indices = torch.Tensor(np.load(pathway_data_path)) # (num of pathways, num of input genes)

        # put datasets on GPU
        pathway_indices, n_pathway_members = pathway_indices.to(device), n_pathway_members.to(device)  
        
        # run model with most optimal hyperparameters
        param_save_path = args.outroot + 'matrix_' + str(i) + '_model_weights'
        hyp_save_path = args.outroot + 'matrix_' + str(i) + '_model_hyp'
        metric_save_path = args.outroot + 'matrix_' + str(i) + '_model_train_metrics'
        
        mkdir(param_save_path)
        mkdir(hyp_save_path)
        mkdir(metric_save_path)
        
        # partition dataset into train and testset
        trainset, testset = prep_data(cl_features, drug_features, label_matrix, 
                                      indices, fold_type=foldtype, 
                                      train_fold=[0,1,2], val_fold=[4])   
    
        y, pred, metric_matrix = train_val(hyp, trainset, testset,
                                        pathway_indices, n_pathway_members,
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

 
  
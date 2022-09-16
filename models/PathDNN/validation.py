# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 23:06:46 2022

@author: jessi


The validation script contains methods for performing train, test, single-fold 
training and validation, and hyperparameter tuning
"""

import torch
import torch.nn as nn
from torch.utils.data import Subset

import time
import numpy as np
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from collections import deque
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from PathDNN.model import PathDNN, PathDNNMLP
from utils.load_data import PathDNNDataset
from utils.utils import set_seed, save_model, load_pretrained_model, mkdir, norm_cl_features, norm_drug_features

def prep_data(cl_features, drug_features, label_matrix, indices, 
              fold_type, train_fold=[0,1,2], val_fold=[3]):
    
    # normalize cl features, no need to normalize drug features since it's
    # all binary feature
    fold = indices[fold_type]
    norm_cl_exp = norm_cl_features(cl_features, indices, 
                                     fold_type, train_fold)
    norm_drug_target = norm_drug_features(drug_features, indices,
                                          fold_type, train_fold)
    # create input data with normalized features
    dataset = PathDNNDataset(norm_cl_exp, norm_drug_target, indices, label_matrix)
    train_idx = np.where(fold.isin(train_fold) == True)[0]
    val_idx = np.where(fold.isin(val_fold) == True)[0]
    trainset = Subset(dataset, train_idx)
    valset = Subset(dataset, val_idx)
    
    return trainset, valset

def train(dataloader, device, model, loss_fn, optimizer, pathway_freeze=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    
    if pathway_freeze == True:
        model.pathway.weight.requires_grad = False
        
    for batch, (X_cl, X_drug, y) in enumerate(dataloader):
        X = torch.cat([X_cl, X_drug],-1)
        X, y = X.to(device), y.to(device)

        # Compute training loss
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    train_loss /= num_batches
    print(f"Train Error: \n Avg loss: {train_loss:>8f} \n")
    return train_loss        
    
    
def test(dataloader, device, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    preds = []
    ys = []
    with torch.no_grad():
        for (X_cl, X_drug, y) in dataloader:
            ys.append(y)
            X = torch.cat([X_cl, X_drug], -1)
            X, y = X.to(device), y.to(device)
            pred = model(X)
#             _pred = torch.squeeze(pred) # shape is (batch_size)
            preds.append(pred)
            # preds.append(pred.squeeze().cpu().detach().numpy())
            # ys.append(y.squeeze().cpu().detach().numpy())

            test_loss += loss_fn(pred, y).item()
            

    preds = torch.cat(preds, axis=0).cpu().detach().numpy().reshape(-1)
    ys = torch.cat(ys, axis=0).reshape(-1)
    
    test_loss /= num_batches
    pearson = pearsonr(ys, preds)[0]
    spearman = spearmanr(ys, preds)[0]
    r2 = r2_score(ys, preds)
    # ys = np.concatenate(ys)
    # preds = np.concatenate(preds)
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss, pearson, spearman, r2, ys, preds





#  performs single-fold train and validation using specified hyperparameters
def train_val(hyp, trainset, valset, pathway_mask, fold_type, model_type='pathdnn', 
              load_pretrain=False, model_path=None, param_save_path=None, 
              hyp_save_path=None, metric_save_path=None, description=None):
    start_time = time.time()
    
    n_in = hyp['n_in']
    n_pathway = hyp['n_pathway']
    n_hidden1 = hyp['n_hidden1']
    n_hidden2 = hyp['n_hidden2']
    # pathway_mask = hyp['pathway_mask']
    drop_rate = hyp['drop_rate']
    batch = hyp['batch_size']
    epoch = hyp['epoch']
    lr_adam = hyp['lr_adam']
    lr_sgd = hyp['lr_sgd']
    decay_adam = hyp['weight_decay_adam']
    decay_sgd = hyp['weight_decay_sgd']
    
    metric_matrix = np.zeros((epoch, 5))
    
    # make model deterministic
    set_seed(0)
    
    # declare device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch, shuffle=False)
    
    # declare model, optimizer and loss function
    if model_type == 'pathdnn':
        model = PathDNN(pathway_mask, n_in, n_pathway, n_hidden1, n_hidden2, drop_rate).to(device)
    elif model_type == 'pathdnn_mlp':
        model = PathDNNMLP(n_in, n_pathway, n_hidden1, n_hidden2, drop_rate).to(device)

        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam, weight_decay=decay_adam) 
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=lr_sgd, weight_decay=decay_sgd)
    loss_fn = nn.MSELoss()
    
    # either load a pre-trained model or train using specified train_fold
    if load_pretrain == True:
        load_pretrained_model(model, model_path)
        
    elif load_pretrain == False:
        # ------------------------ training begins --------------------------------------
        for t in range(epoch):
            if t > int(epoch * 0.9):
                optimizer = optimizer_sgd
            train_loss = train(train_loader, device, model, loss_fn, optimizer, pathway_freeze=False)
            test_loss, pearson, spearman, r2, ys, preds = test(val_loader, device, model, loss_fn)
            metric_matrix[t, 0] = train_loss
            metric_matrix[t, 1:] = [test_loss, pearson, spearman, r2]
            print("epoch:%d\ttrain-mse:%.4f\tval-mse:%.4f\tval-pcc:%.4f\tval-spc:%.4f\tval-r2:%.4f"%(
                    t, train_loss, test_loss, pearson, spearman, r2))
            
        # --------------------- save trained model parameters ---------------------------
        save_model(model, hyp, metric_matrix,
                   param_save_path, hyp_save_path, metric_save_path, description)
        
    
    # ------------------------ testing begins ---------------------------------------
    test_loss, pearson, spearman, r2, ys, preds = test(val_loader, device, model, loss_fn)
    elapsed_time = time.time() - start_time
    print("Done " + fold_type + " single-fold train and validation")
    print("val-mse:%.4f\tval-pcc:%.4f\tval-spc:%.4f\tval-r2:%.4f\t%ds"%(
            test_loss, pearson, spearman, r2, int(elapsed_time)))
    return ys, preds, metric_matrix



def hyper_tune_train(config, trainset, valset, fold_type, pathway_mask, num_epoch,
                     model_type='pathdnn', description=None, save_path=None):
    
    start_time = time.time()
    # num_epoch = 100
    metric_matrix = np.zeros((num_epoch, 5))
    metric_names = ['train_MSE', 'val_MSE', 'val_PCC', 'val_SPC', 'val_R2']
    loss_deque = deque([], maxlen=5)

    best_loss = np.inf
    best_loss_avg5 = np.inf
    best_loss_epoch = 0
    best_avg5_loss_epoch = 0
    
    n_in = config['n_in']
    n_pathway = config['n_pathway']
    # pathway_mask = config['pathway_mask']
    
    # make model deterministic
    set_seed(0)
    
    # declare device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=False)
    
    # ----------------- declare model, opt_fn, loss_fn ------------------------
    if model_type == 'pathdnn':
        model = PathDNN(pathway_mask, n_in, n_pathway, 
                        config['n_hidden1'], config['n_hidden2'], config['drop_rate']).to(device)
    elif model_type == 'pathdnn_mlp':
        model = PathDNNMLP(n_in, n_pathway, 
                           config['n_hidden1'], config['n_hidden2'], config['drop_rate']).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_adam'], 
                                 weight_decay=config['weight_decay_adam']) 
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=config['lr_sgd'], 
                                    weight_decay=config['weight_decay_sgd'])
    loss_fn = nn.MSELoss()
    
    
    for epoch in range(num_epoch):
        if epoch > int(num_epoch*0.9):
            optimizer = optimizer_sgd
        # ------------------------ training begins ----------------------------
        train_loss = train(train_loader, device, model, loss_fn, optimizer)
        
        # --------------------- validation begins -----------------------------
        test_loss, pearson, spearman, r2, ys, preds = test(val_loader, device, model, loss_fn)
        
        metric_matrix[epoch, 0] = train_loss
        metric_matrix[epoch, 1:] = [test_loss, pearson, spearman, r2]
        
        if best_loss > test_loss:
            best_loss = test_loss
            best_loss_epoch = epoch + 1
        
        loss_deque.append(test_loss)
        loss_avg5 = sum(loss_deque)/len(loss_deque)
        
        if best_loss_avg5 > loss_avg5:
            best_loss_avg5 = loss_avg5
            best_avg5_loss_epoch = epoch + 1 
        
        # communicate with Ray Tune
        tune.report(loss=test_loss, pcc=pearson, spc=spearman, r2=r2,
                    best_loss_epoch=best_loss_epoch, 
                    best_avg5_loss_epoch=best_avg5_loss_epoch)


        elapsed_time = time.time() - start_time
        start_time = time.time()
        print("%d\ttrain-mse:%.4f\tval-mse:%.4f\tval-pcc:%.4f\tval-spc:%.4f\tval-r2:%.4f\t%ds"%(
                epoch, train_loss, test_loss, pearson, spearman, r2, int(elapsed_time)))
    return metric_matrix, metric_names
    
# hyper_tune_train(num_epoch, config, hyp, trainset, valset, fold_type, 
#                      model_type='pathdnn', description=None, save_path=None)

def hyper_tune_main(trainset, valset, fold_type, grid, pathway_mask,
                    num_samples, num_train_epoch,
                    model_type = 'pathdnn', save_path=None):
    ray.init(ignore_reinit_error=True)
    config = {
        'n_in': grid['n_in'],
        'n_pathway': grid['n_pathway'],
        'n_hidden1': tune.choice(grid['n_hidden1']),
        'n_hidden2': tune.choice(grid['n_hidden2']),
        'drop_rate': tune.choice(grid['drop_rate']),
        'batch_size': tune.choice(grid['batch_size']),
        'lr_adam': tune.grid_search(grid['lr_adam']),
        'weight_decay_adam': tune.grid_search(grid['weight_decay_adam']),
        'lr_sgd': tune.grid_search(grid['lr_sgd']),
        'weight_decay_sgd': tune.grid_search(grid['weight_decay_sgd'])
        }
    
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        # metric='loss',
        # mode='min',
        max_t=100,
        grace_period=10,
        reduction_factor=3
        )
    
    reporter = CLIReporter(
        max_progress_rows=5,
        print_intermediate_tables=False,
        metric_columns=["loss", "pcc", "best_loss_epoch", "best_avg5_loss_epoch", "training_iteration"])
    

    result = tune.run(
        tune.with_parameters(hyper_tune_train, 
                             trainset=trainset, valset=valset, 
                             fold_type=fold_type, pathway_mask=pathway_mask,
                             num_epoch=num_train_epoch,
                             model_type=model_type),
        name = 'pathdnn',
        metric='pcc',
        mode='max',
        resources_per_trial={"cpu": 2, "gpu": 0.33},
        config = config,
        verbose=3,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        reuse_actors=True
        )
    
    best_trial = result.get_best_trial('pcc', 'max', 'all')
    epoch = best_trial.last_result["best_loss_epoch"]
    epoch_avg5 = best_trial.last_result["best_avg5_loss_epoch"]
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation PCC: {}".format(best_trial.last_result["pcc"]))
    print("Best trail epoch: {}".format(epoch))
    print("Best trail avg5_epoch: {}".format(epoch_avg5))
    ray.shutdown()
    return epoch, epoch_avg5, best_trial.config



























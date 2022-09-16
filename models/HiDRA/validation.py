# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:45:10 2022

@author: jessi
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

from model import HiDRA
from utils.load_data import HiDRADataset
from utils.utils import set_seed, save_model, load_pretrained_model, mkdir, norm_cl_features

def prep_data(cl_features, drug_features, label_matrix, indices, 
              fold_type, train_fold=[0,1,2], val_fold=[3]):
    
    # normalize cl features, no need to normalize drug features since it's
    # all binary feature
    fold = indices[fold_type]
    norm_cl_exp = norm_cl_features(cl_features, indices, 
                                     fold_type, train_fold)
    
    # create input data with normalized features
    dataset = HiDRADataset(norm_cl_exp, drug_features, indices, label_matrix)
    train_idx = np.where(fold.isin(train_fold) == True)[0]
    val_idx = np.where(fold.isin(val_fold) == True)[0]
    trainset = Subset(dataset, train_idx)
    valset = Subset(dataset, val_idx)
    
    return trainset, valset

def train(dataloader, device, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0

    for batch, (X_cl, X_drug, y) in enumerate(dataloader):
        X_cl, X_drug, y = X_cl.to(device), X_drug.to(device), y.to(device)

        # Compute training loss
        pred = model(X_drug, X_cl)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(X_drug)
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
            # X = torch.cat([X_cl, X_drug], -1)
            X_cl, X_drug, y = X_cl.to(device), X_drug.to(device), y.to(device)
            pred = model(X_drug, X_cl)
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



def train_val(hyp, trainset, valset, pathway_indices, n_member_genes, 
              fold_type, model_type='hidra', 
              load_pretrain=False, model_path=None, param_save_path=None, 
              hyp_save_path=None, metric_save_path=None, description=None):
    start_time = time.time()
    

    batch = hyp['batch_size']
    epoch = hyp['epoch']
    lr_adam = hyp['lr_adam']
    decay_adam = hyp['weight_decay_adam']
    
    metric_matrix = np.zeros((epoch, 5))
    loss_deque = deque([], maxlen=5)

    best_loss = np.inf
    best_loss_avg5 = np.inf
    best_loss_epoch = 0
    best_avg5_loss_epoch = 0
    
    # make model deterministic
    set_seed(0)
    
    # declare device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch, shuffle=False)
    
    # declare model, optimizer and loss function
    if model_type == 'hidra':
        model = HiDRA(hyp, pathway_indices, n_member_genes).to(device)
    elif model_type == 'hidra_mlp':
        model = None
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam, weight_decay=decay_adam) 
    loss_fn = nn.MSELoss()
    
    # either load a pre-trained model or train using specified train_fold
    if load_pretrain == True:
        load_pretrained_model(model, model_path)
        
        # ------------------------ testing begins ---------------------------------------
        test_loss, pearson, spearman, r2, ys, preds = test(val_loader, device, model, loss_fn)
        elapsed_time = time.time() - start_time
        print("Done " + fold_type + " single-fold train and validation")
        print("val-mse:%.4f\tval-pcc:%.4f\tval-spc:%.4f\tval-r2:%.4f\t%ds"%(
                test_loss, pearson, spearman, r2, int(elapsed_time)))
        return ys, preds, metric_matrix
    
    elif load_pretrain == False:
        # ------------------------ training begins --------------------------------------
        for t in range(epoch):
            train_loss = train(train_loader, device, model, loss_fn, optimizer)
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


def hyper_tune_train(config, trainset, valset, 
                     pathway_indices, n_member_genes,
                     fold_type, num_epoch,
                     model_type='hidra', description=None, save_path=None):
    
    start_time = time.time()
    metric_matrix = np.zeros((num_epoch, 5))
    metric_names = ['train_MSE', 'val_MSE', 'val_PCC', 'val_SPC', 'val_R2']
    loss_deque = deque([], maxlen=5)

    best_loss = np.inf
    best_loss_avg5 = np.inf
    best_loss_epoch = 0
    best_avg5_loss_epoch = 0
    
    
    # make model deterministic
    set_seed(0)
    
    # declare device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=False)
    
    # ----------------- declare model, opt_fn, loss_fn ------------------------
    if model_type == 'hidra':
        model = HiDRA(config, pathway_indices, n_member_genes).to(device)
    elif model_type == 'hidra_mlp':
        model = None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_adam'], 
                                 weight_decay=config['weight_decay_adam']) 

    loss_fn = nn.MSELoss()
    
    
    for epoch in range(num_epoch):
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




def hyper_tune_main(trainset, valset, pathway_indices, n_member_genes, grid,
                    fold_type, num_samples, num_train_epoch,
                    model_type = 'hidra', save_path=None):
    ray.init(ignore_reinit_error=True)
    config = {
        'n_drug_feature': grid['n_drug_feature'], 
        'n_gene_feature': grid['n_gene_feature'], 
        'n_pathway': grid['n_pathway'],
        'n_drugnet1': grid['n_drugnet1'],
        'n_drugnet2': grid['n_drugnet2'],
        'n_drugatt1': tune.choice(grid['n_drugatt1']),
        'n_drugatt2': tune.choice(grid['n_drugatt2']),   
        'n_drugenc_gene': tune.choice(grid['n_drugenc_gene']),
        'n_drugenc_pathway': tune.choice(grid['n_drugenc_pathway']), 
        'batch_size': tune.choice(grid['batch_size']), 
        'lr_adam': tune.grid_search(grid['lr_adam']), 
        'weight_decay_adam': 0
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
        tune.with_parameters(hyper_tune_train, trainset=trainset, valset=valset, 
                             fold_type=fold_type, num_epoch=num_train_epoch,
                             pathway_indices=pathway_indices, 
                             n_member_genes=n_member_genes,
                             model_type=model_type),
        name = 'hidra',
        metric='loss',
        mode='min',
        resources_per_trial={"cpu": 2, "gpu": 0.33}, #2, 0.33
        config = config,
        verbose=3,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        reuse_actors=True
        )
    
    best_trial = result.get_best_trial('loss', 'min', 'all')
    epoch = best_trial.last_result["best_loss_epoch"]
    epoch_avg5 = best_trial.last_result["best_avg5_loss_epoch"]
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trail final validation PCC: {}".format(best_trial.last_result["pcc"]))
    print("Best trail epoch: {}".format(epoch))
    print("Best trail avg5_epoch: {}".format(epoch_avg5))
    ray.shutdown()
    return epoch, epoch_avg5, best_trial.config


















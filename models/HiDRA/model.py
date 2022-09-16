# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:41:12 2022

@author: jessi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)
        self.iter = 0

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)
    
class DrugNetwork(nn.Module):
    def __init__(self, n_drug_in, n_hidden1, n_hidden2):
        super(DrugNetwork, self).__init__()
        self.layer1 = nn.Linear(n_drug_in, n_hidden1)
        self.batchNorm1 = nn.BatchNorm1d(n_hidden1)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.batchNorm2 = nn.BatchNorm1d(n_hidden2)
        
    def forward(self, x):
        x = torch.relu(self.batchNorm1(self.layer1(x)))
        x = torch.relu(self.batchNorm2(self.layer2(x)))
        return x

class DrugAttentionNetwork(nn.Module):
    def __init__(self, n_drug_in, n_hidden1, n_hidden2):
        super(DrugAttentionNetwork, self).__init__()
        self.layer1 = nn.Linear(n_drug_in, n_hidden1)
        self.batchNorm1 = nn.BatchNorm1d(n_hidden1)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.batchNorm2 = nn.BatchNorm1d(n_hidden2)
        
    def forward(self, x):
        x = torch.relu(self.batchNorm1(self.layer1(x)))
        x = torch.relu(self.batchNorm2(self.layer2(x)))
        return x


class GeneNetwork(nn.Module):
    def __init__(self, n_drug_att_in, n_drug1, n_gene_in):
        super(GeneNetwork, self).__init__()
        self.drug_input = nn.Linear(n_drug_att_in, n_drug1)
        self.batchNormDrug = nn.BatchNorm1d(n_drug1)
        # self.gene_input = nn.MaskedLinear(n_gene_in)
        self.gene_att = nn.Linear(n_drug1+n_gene_in, n_gene_in)
        self.batchNormDrugGene = nn.BatchNorm1d(1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, drug_att, genes):
        drug_att = torch.relu(self.batchNormDrug(self.drug_input(drug_att)))
        # print(1, drug_att.shape) #128, 11
        if genes.dim() == 1:
            genes = genes.unsqueeze(dim=-1)
        drug_att_genes = torch.cat((genes, drug_att), dim=-1)
        # print(2, drug_att_genes.shape) #128, 52
        drug_att_genes = torch.tanh(self.gene_att(drug_att_genes))
        # print(3, drug_att_genes.shape) #128, 41
        drug_att_genes = self.softmax(drug_att_genes) # dim: n_gene_in
        # print(4, drug_att_genes.shape) #128, 41
        drug_att_genes = (genes * drug_att_genes).sum(dim=-1).unsqueeze(dim=-1) # batch_size * 1
        # print(5, drug_att_genes.shape) #128
        drug_att_genes = torch.relu(self.batchNormDrugGene(drug_att_genes))
        # print(6, drug_att_genes.shape)
        return drug_att_genes

class PathwayNetwork(nn.Module):
    def __init__(self, n_drug_att_in, n_drug1, n_drug_att_genes, n_pathway):
        super(PathwayNetwork, self).__init__()
        self.drug_input = nn.Linear(n_drug_att_in, n_drug1)
        self.batchNormDrug = nn.BatchNorm1d(n_drug1)
        self.pathway_att = nn.Linear(n_drug1+n_drug_att_genes, n_pathway)
        self.softmax = nn.Softmax(dim=1)
        self.batchNormPathway = nn.BatchNorm1d(n_pathway)
    
    def forward(self, drug_att, attention_dot): # attention dot dim = n_pathway
        drug_att = torch.relu(self.batchNormDrug(self.drug_input(drug_att)))
        drug_att_pathway = torch.cat((attention_dot,drug_att), dim=-1)
        drug_att_pathway = torch.tanh(self.pathway_att(drug_att_pathway))
        drug_att_pathway = self.batchNormPathway(drug_att_pathway * attention_dot)
        drug_att_pathway = torch.relu(drug_att_pathway)
        return drug_att_pathway


class HiDRAOutput(nn.Module):
    def __init__(self, hyp):
        super(HiDRAOutput, self).__init__()
        # need to add 128b/c dim of drug embedding is 128
        self.layer1 = nn.Linear(hyp['n_drugnet2']+hyp['n_pathway'], 128)
        self.batchNorm1 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.batchNorm1(self.layer1(x)))
        x = self.output(x)
        return x
    
    
    
    
class HiDRA(nn.Module):
    def __init__(self, hyp, pathway_indices, n_pathway_members):
        super(HiDRA, self).__init__()
        self.pathway_indices = pathway_indices # A PyTorch Tensor, dim = (n_pathway, n_genes)
        
        self.n_pathway_members = n_pathway_members
        
        self.drugNetwork = DrugNetwork(hyp['n_drug_feature'], #512
                                       hyp['n_drugnet1'], #256
                                       hyp['n_drugnet2']) #128
        
        self.drugAttNetwork = DrugAttentionNetwork(hyp['n_drug_feature'], #512
                                                   hyp['n_drugatt1'], #128
                                                   hyp['n_drugatt2']) #32
        
        self.geneNetworks = nn.ModuleList([GeneNetwork(hyp['n_drugatt2'], #32
                                       int(self.n_pathway_members[i]/hyp['n_drugenc_gene'] + 1), # len(member genes)/4 + 1
                                       int(self.n_pathway_members[i])) \
                                      for i in range(hyp['n_pathway'])])
            
        self.pathwayNetwork = PathwayNetwork(hyp['n_drugatt2'], #32
                                             int(hyp['n_pathway']/hyp['n_drugenc_pathway'] + 1), # num_pathways/16  + 1 
                                             hyp['n_pathway'],
                                             hyp['n_pathway'])
            
        self.HiDRA_output = HiDRAOutput(hyp)
        
    
    def forward(self, drug_features, cl_features):
        drug_embed = self.drugNetwork(drug_features)        # (batch_size, 128)
        drug_att_embed = self.drugAttNetwork(drug_features) # (batch_size, 32)
        gene_att_dot_ls = [] 
        # print(drug_embed.shape)
        # print(drug_att_embed.shape)
        for i, geneNet in enumerate(self.geneNetworks):
            # In each iter, the gene_att_dot corresponds to a specific 
            # pathway, therefore the result from each iter is saved to a 
            # list and concatenated afterwards to be fed into the 
            # pathway attention network
            # print(self.pathway_indices[i].shape) #5511
            # print(cl_features.shape) #[128, 5511]
            member_genes = self.pathway_indices[i] * cl_features # (batch_size, 5511)
            # print(member_genes.shape)
            member_genes = member_genes[:, self.pathway_indices[i].nonzero()].squeeze() # (batch_size, # of member genes)
            # print(drug_att_embed.shape) #(128, 32)
            # print(member_genes.shape) #(128, 41)
            
            gene_att_dot = geneNet(drug_att_embed, member_genes) # (batch_size, 1)
            # print('gene_att_dot shape: ', gene_att_dot.shape)
            gene_att_dot_ls.append(gene_att_dot) # list size: n_pathway x 1
        
        # convert gene_att_dot_ls to a PyTorch Tensor
        gene_att_dot_tensor = torch.transpose(torch.stack(gene_att_dot_ls), 0, 1)
        gene_att_dot_tensor = gene_att_dot_tensor.squeeze()
        # print('gene_att_dot_tensor shape:', gene_att_dot_tensor.shape) # [332, batch_sze, 1]
        pathway_att = self.pathwayNetwork(drug_att_embed, gene_att_dot_tensor) # dim = n_pathway
        
        # concatenate drug_embeddings and pathway_att together
        hidra_x = torch.cat((drug_embed, pathway_att), dim=-1)
        hidra_x = self.HiDRA_output(hidra_x)
        return hidra_x
        

        





















#  -------------- OLD CODE before hyperparam tuning -------------------------------------
# # -*- coding: utf-8 -*-
# """
# Created on Mon Mar 28 17:20:29 2022

# @author: jessi
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class MaskedLinear(nn.Linear):
#     def __init__(self, in_features, out_features, mask, bias=True):
#         super(MaskedLinear, self).__init__(in_features, out_features, bias)
#         self.register_buffer('mask', mask)
#         self.iter = 0

#     def forward(self, input):
#         masked_weight = self.weight * self.mask
#         return F.linear(input, masked_weight, self.bias)
    
# class DrugNetwork(nn.Module):
#     def __init__(self, n_drug_in, n_hidden1, n_hidden2):
#         super(DrugNetwork, self).__init__()
#         self.layer1 = nn.Linear(n_drug_in, n_hidden1)
#         self.batchNorm1 = nn.BatchNorm1d(n_hidden1)
#         self.layer2 = nn.Linear(n_hidden1, n_hidden2)
#         self.batchNorm2 = nn.BatchNorm1d(n_hidden2)
        
#     def forward(self, x):
#         x = torch.relu(self.batchNorm1(self.layer1(x)))
#         x = torch.relu(self.batchNorm2(self.layer2(x)))
#         return x

# class DrugAttentionNetwork(nn.Module):
#     def __init__(self, n_drug_in, n_hidden1, n_hidden2):
#         super(DrugAttentionNetwork, self).__init__()
#         self.layer1 = nn.Linear(n_drug_in, n_hidden1)
#         self.batchNorm1 = nn.BatchNorm1d(n_hidden1)
#         self.layer2 = nn.Linear(n_hidden1, n_hidden2)
#         self.batchNorm2 = nn.BatchNorm1d(n_hidden2)
        
#     def forward(self, x):
#         x = torch.relu(self.batchNorm1(self.layer1(x)))
#         x = torch.relu(self.batchNorm2(self.layer2(x)))
#         return x


# class GeneNetwork(nn.Module):
#     def __init__(self, n_drug_att_in, n_drug1, n_gene_in):
#         super(GeneNetwork, self).__init__()
#         self.drug_input = nn.Linear(n_drug_att_in, n_drug1)
#         self.batchNormDrug = nn.BatchNorm1d(n_drug1)
#         # self.gene_input = nn.MaskedLinear(n_gene_in)
#         self.gene_att = nn.Linear(n_drug1+n_gene_in, n_gene_in)
#         self.batchNormDrugGene = nn.BatchNorm1d(1)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, drug_att, genes):
#         drug_att = torch.relu(self.batchNormDrug(self.drug_input(drug_att)))
#         # print(1, drug_att.shape) #128, 11
#         drug_att_genes = torch.cat((genes, drug_att), dim=-1)
#         # print(2, drug_att_genes.shape) #128, 52
#         drug_att_genes = torch.tanh(self.gene_att(drug_att_genes))
#         # print(3, drug_att_genes.shape) #128, 41
#         drug_att_genes = self.softmax(drug_att_genes) # dim: n_gene_in
#         # print(4, drug_att_genes.shape) #128, 41
#         drug_att_genes = (genes * drug_att_genes).sum(dim=-1).unsqueeze(dim=-1) # batch_size * 1
#         # print(5, drug_att_genes.shape) #128
#         drug_att_genes = torch.relu(self.batchNormDrugGene(drug_att_genes))
#         # print(6, drug_att_genes.shape)
#         return drug_att_genes

# class PathwayNetwork(nn.Module):
#     def __init__(self, n_drug_att_in, n_drug1, n_drug_att_genes, n_pathway):
#         super(PathwayNetwork, self).__init__()
#         self.drug_input = nn.Linear(n_drug_att_in, n_drug1)
#         self.batchNormDrug = nn.BatchNorm1d(n_drug1)
#         self.pathway_att = nn.Linear(n_drug1+n_drug_att_genes, n_pathway)
#         self.softmax = nn.Softmax(dim=1)
#         self.batchNormPathway = nn.BatchNorm1d(n_pathway)
    
#     def forward(self, drug_att, attention_dot): # attention dot dim = n_pathway
#         drug_att = torch.relu(self.batchNormDrug(self.drug_input(drug_att)))
#         drug_att_pathway = torch.cat((attention_dot,drug_att), dim=-1)
#         drug_att_pathway = torch.tanh(self.pathway_att(drug_att_pathway))
#         drug_att_pathway = self.batchNormPathway(drug_att_pathway * attention_dot)
#         drug_att_pathway = torch.relu(drug_att_pathway)
#         return drug_att_pathway


# class HiDRAOutput(nn.Module):
#     def __init__(self, hyp):
#         super(HiDRAOutput, self).__init__()
#         # need to add 128b/c dim of drug embedding is 128
#         self.layer1 = nn.Linear(128+hyp['n_pathway'], 128)
#         self.batchNorm1 = nn.BatchNorm1d(128)
#         self.output = nn.Linear(128, 1)
    
#     def forward(self, x):
#         x = torch.relu(self.batchNorm1(self.layer1(x)))
#         x = self.output(x)
#         return x
    
    
    
    
# class HiDRA(nn.Module):
#     def __init__(self, hyp, pathway_indices, n_pathway_members):
#         super(HiDRA, self).__init__()
#         self.pathway_indices = pathway_indices # A PyTorch Tensor, dim = (n_pathway, n_genes)
        
#         self.n_pathway_members = n_pathway_members
        
#         self.drugNetwork = DrugNetwork(hyp['n_drug_feature'], #512
#                                        256, #hyp['n_drugnet1']
#                                        128) #hyp['n_drugnet2']
        
#         self.drugAttNetwork = DrugAttentionNetwork(hyp['n_drug_feature'], #512
#                                                    128, #hyp['n_drugattnet1']
#                                                    32) #hyp['n_drugattnet2']
        
#         self.geneNetworks = nn.ModuleList([GeneNetwork(32, #hyp['n_drugattnet2']
#                                        int(self.n_pathway_members[i]/4 + 1), # len(member genes)/4 + 1
#                                        int(self.n_pathway_members[i])) \
#                                       for i in range(hyp['n_pathway'])])
            
#         self.pathwayNetwork = PathwayNetwork(32, #hyp['n_drugattnet2']
#                                              int(hyp['n_pathway']/16 + 1), # num_pathways/16  + 1 
#                                              hyp['n_pathway'],
#                                              hyp['n_pathway'])
            
#         self.HiDRA_output = HiDRAOutput(hyp)
        
    
#     def forward(self, drug_features, cl_features):
#         drug_embed = self.drugNetwork(drug_features)        # (batch_size, 128)
#         drug_att_embed = self.drugAttNetwork(drug_features) # (batch_size, 32)
#         gene_att_dot_ls = [] 
#         # print(drug_embed.shape)
#         # print(drug_att_embed.shape)
#         for i, geneNet in enumerate(self.geneNetworks):
#             # In each iter, the gene_att_dot corresponds to a specific 
#             # pathway, therefore the result from each iter is saved to a 
#             # list and concatenated afterwards to be fed into the 
#             # pathway attention network
#             # print(self.pathway_indices[i].shape) #5511
#             # print(cl_features.shape) #[128, 5511]
#             member_genes = self.pathway_indices[i] * cl_features # (batch_size, 5511)
#             # print(member_genes.shape)
#             member_genes = member_genes[:, self.pathway_indices[i].nonzero()].squeeze() # (batch_size, # of member genes)
#             # print(drug_att_embed.shape) #(128, 32)
#             # print(member_genes.shape) #(128, 41)
            
#             gene_att_dot = geneNet(drug_att_embed, member_genes) # (batch_size, 1)
#             # print('gene_att_dot shape: ', gene_att_dot.shape)
#             gene_att_dot_ls.append(gene_att_dot) # list size: n_pathway x 1
        
#         # convert gene_att_dot_ls to a PyTorch Tensor
#         gene_att_dot_tensor = torch.transpose(torch.stack(gene_att_dot_ls), 0, 1)
#         gene_att_dot_tensor = gene_att_dot_tensor.squeeze()
#         # print('gene_att_dot_tensor shape:', gene_att_dot_tensor.shape) # [332, batch_sze, 1]
#         pathway_att = self.pathwayNetwork(drug_att_embed, gene_att_dot_tensor) # dim = n_pathway
        
#         # concatenate drug_embeddings and pathway_att together
#         hidra_x = torch.cat((drug_embed, pathway_att), dim=-1)
#         hidra_x = self.HiDRA_output(hidra_x)
#         return hidra_x
        
        



#         return x
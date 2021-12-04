# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:43:17 2021

@author: giuli
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# path to preprocessed dataset
path_preproc = '..\reionisation_ML\dataset'

# number of data to use in the training and validation
dataset_size = 2000

# load and prepare dataset with shape (dataset_size, input_type, channel_size, xdim, ydim, zdim)
X = np.zeros((dataset_size, 2, 1, 49, 49, 49))
for i in range(dataset_size):
    n_src = np.load('%sn_src_i%d.npy' %(path_preproc, i))
    n_igm = np.load('%sn_igm_i%d.npy' %(path_preproc, i))
    X[i,0] = n_src[np.newaxis, ...]
    X[i,1] = n_igm[np.newaxis, ...]
y = np.loadtxt('%sxi_flatten.txt' %path_preproc)[:dataset_size]

# split dataset into trianing (80%) and validation set (test_size = 20%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2021)

# convert numpy array to torch tensor
X_train_src, X_train_igm = torch.Tensor(X_train[:,0,:,:,:,:]), torch.Tensor(X_train[:,1,:,:,:,:])
X_valid_src, X_valid_igm = torch.Tensor(X_valid[:,0,:,:,:,:]), torch.Tensor(X_valid[:,1,:,:,:,:])

y_train = torch.Tensor(y_train)
y_valid = torch.Tensor(y_valid)

# create pytorch dataset
train_dataset = TensorDataset(X_train_src, X_train_igm, y_train)
valid_dataset = TensorDataset(X_valid_src, X_valid_igm, y_valid)

# create data loader to use in epoch for loop, for a batch of size 32
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
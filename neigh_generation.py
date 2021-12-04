# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:06:26 2021

@author: giuli
"""

import numpy as np
import torch
from torch import nn
from torch import optim
import random
import itertools

def get_neighborhood(T,P,r): # T = original tensor, (x0,y0,z0) central point, r = radius 
    
    edge = T.shape[0]
    
    x0 = P[0]
    y0 = P[1]
    z0 = P[2]
    
    # '+' for lists is to concatenate when x0+r exceeds the cube edge
    idx = list(range(x0-r, min(edge,x0+r+1))) + list(range(max(edge,x0+r+1) % edge))
    idy = list(range(y0-r, min(edge,y0+r+1))) + list(range(max(edge,y0+r+1) % edge))
    idz = list(range(z0-r, min(edge,z0+r+1))) + list(range(max(edge,z0+r+1) % edge))
    
    Tm = T[:,:,:]
    neigh = Tm[idx,:,:][:,idy,:][:,:,idz]
    neigh = np.expand_dims(neigh, axis=0)
    neigh = np.expand_dims(neigh, axis=0)
    return neigh

    
z = 8.397
r = 24
    
n_igm = np.load('../dataset/rho_z%.3f.npy' %z)  # density of intergalactic medium\n"
n_src = np.load('../dataset/nsrc_z%.3f.npy' %z) # number of sources per volume\n"

dims = n_igm.shape

total_points = list(itertools.product(range(dims[0]),range(dims[1]),range(dims[2]))) # cartesian product

small_total = total_points[:10]

for count,P in enumerate(small_total):

    n_igm_nbh = torch.tensor(get_neighborhood(n_igm, P, r)).float()
    n_src_nbh = torch.tensor(get_neighborhood(n_src, P, r)).float()
    
    np.save('n_igm_i%d.npy' % count, n_igm_nbh)
    np.save('n_src_i%d.npy' % count, n_src_nbh)
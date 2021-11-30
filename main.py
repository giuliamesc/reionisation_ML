import numpy as np
import CNN
import torch
from torch import nn
from torch import optim
import random
import itertools




def get_neighborhood(T,P,r): # T = original tensor, (x0,y0,z0) central point, r = radius 
    
    edge = T.shape[3]
    
    x0 = P[0]
    y0 = P[1]
    z0 = P[2]
    
    # '+' for lists is to concatenate when x0+r exceeds the cube edge
    idx = list(range(x0-r, min(edge,x0+r+1))) + list(range(max(edge,x0+r+1) % edge))
    idy = list(range(y0-r, min(edge,y0+r+1))) + list(range(max(edge,y0+r+1) % edge))
    idz = list(range(z0-r, min(edge,z0+r+1))) + list(range(max(edge,z0+r+1) % edge))
    
    Tm = T[0,0,:,:,:]
    neigh = Tm[idx,:,:][:,idy,:][:,:,idz]
    neigh = np.expand_dims(neigh, axis=0)
    neigh = np.expand_dims(neigh, axis=0)
    return neigh



def preprocessing(n_igm,n_src):
    
    n_igm = 2.95 * 10**55 * n_igm
    mean_n_igm = np.mean(n_igm)
    std_n_igm = np.std(n_igm)
    mean_n_src = np.mean(n_src)
    std_n_src = np.std(n_src)

    n_igm = (n_igm - mean_n_igm) / std_n_igm
    n_src = (n_src - mean_n_src) / std_n_src
    
    n_igm = np.expand_dims(n_igm, axis=0)
    n_igm = np.expand_dims(n_igm, axis=0)
    n_src = np.expand_dims(n_src, axis=0)
    n_src = np.expand_dims(n_src, axis=0)
    
    return n_igm, n_src



def printing (loss, epoch, n_epochs, iter, n_iters):
    print ('epoch ', epoch+1,'/',n_epochs, '   iter ',iter+1, '/',n_iters, '      loss = ', torch.Tensor.detach(loss).item())
    


    
# Main file

if __name__ == '__main__':
    
    # DATA IMPORT 
    
    z = 8.397
    r = 24
    epochs = 10
    
    n_igm = np.load('../dataset/rho_z%.3f.npy' %z)  # density of intergalactic medium\n"
    n_src = np.load('../dataset/nsrc_z%.3f.npy' %z) # number of sources per volume\n"
    xi = np.load('../dataset/xHII_z%.3f.npy' %z) # ionization fraction"
    
    
    dims = n_igm.shape
    
    # PREPROCESSING
    n_igm, n_src = preprocessing(n_igm, n_src)
    
    
    # CNN CREATION
    net = CNN.CNN()
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    
    
    
    
    
    # TRAINING
    
    random.seed(2021)
    total_points = list(itertools.product(range(dims[0]),range(dims[1]),range(dims[2]))) # cartesian product
    
    
    for epoch in range(epochs):
        
        batch = random.sample(total_points, k = 3000)
        batch_tr = batch[:2500]
        batch_te = batch[2500:]
      
        for iter,P in enumerate(batch_tr):
                
            n_igm_nbh = torch.tensor(get_neighborhood(n_igm, P, r)).float()
            n_src_nbh = torch.tensor(get_neighborhood(n_src, P, r)).float()
            
            loss_fn = torch.nn.MSELoss()
            optimizer.zero_grad()  # set the gradients to 0
            output= net(n_igm_nbh, n_src_nbh) # forward
            estimation = torch.tensor([xi[P]]).float()
            loss = loss_fn(output, estimation)  # compute loss function
            printing (loss, epoch, epochs, iter, 2500)
            loss.backward()  # backpropagation
            optimizer.step()


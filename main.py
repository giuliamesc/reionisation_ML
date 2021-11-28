# Main file

# Package loading
import numpy as np
import torch
from torch import nn
from torch import optim

# Extracting the neighborhood (of radius r) of the point with coordinates (x0,y0,z0)

def get_neighborhood(T,x0,y0,z0,r): # T = original tensor, (x0,y0,z0) central point, r = radius 
    
    idx = range(x0-r,x0+r+1)
    idy = range(y0-r,y0+r+1)
    idz = range(z0-r,z0+r+1)
    neigh = T[idx,:,:][:,idy,:][:,:,idz]
    neigh = torch.from_numpy(neigh)
    neigh = neigh[None, None, :,:,:]
    return neigh


if __name__ == '__main__':
    
    # DATA IMPORT 
    
    z = 8.397
    r = 24
    
    n_igm = np.load('rho_z%.3f.npy' %z)  # density of intergalactic medium\n"
    n_src = np.load('nsrc_z%.3f.npy' %z) # number of sources per volume\n"
    xi = np.load('xHII_z%.3f.npy' %z) # ionization fraction"
    dims = n_igm.shape

    # PREPROCESSING
    
    n_igm = 2.95 * 10**55 * n_igm
    mean_n_igm = np.mean(n_igm)
    std_n_igm = np.std(n_igm)
    mean_n_src = np.mean(n_src)
    std_n_src = np.std(n_src)

    n_igm_norm = (n_igm - mean_n_igm) / std_n_igm
    n_src_norm = (n_src - mean_n_src) / std_n_src
    
    net = CNN()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    epochs = 10

    output_matrix = np.zeros(dims)
    
    # TRAINING
    
    for epoch in range(epochs):
      
        for x in range(dims[0]):
            for y in range(dims[1]):
                for z in range(dims[2]):
                
                    n_igm_nbh = get_neighborhood(n_igm, x,y,z, r)
                    n_src_nbh = get_neighborhood(n_src, x,y,z, r)
                    loss_fn = torch.nn.MSELoss()
                    optimizer.zero_grad()  # set the gradients to 0
                    output= net(n_igm_nbh.float(), n_src_nbh.float())
                    temp = torch.Tensor.detach(output)
                    to_output_matrix = torch.Tensor.numpy(temp)
                    output_matrix[x,y,z] = to_output_matrix
                    print(output.shape)
                    estimation = torch.Tensor([xi[x,y,z]])
                    print(estimation)
                    loss = loss_fn(output, estimation)  # compute loss function
                    loss.backward()  # backpropagation
                    optimizer.step()


    print("Epoch {:.2f} | Training loss: {:.5f}".format(epoch, loss))


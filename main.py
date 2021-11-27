import numpy as np
import CNN
from torch import nn
from torch import optim




def get_neighborhood(T,x0,y0,z0,r): # T = original tensor, (x0,y0,z0) central point, r = radius 
    
    idx = range(x0-r,x0+r+1)
    idy = range(y0-r,y0+r+1)
    idz = range(z0-r,z0+r+1)
    neigh = T[idx,:,:][:,idy,:][:,:,idz]
    return neigh


    
if __name__ == '__main__':
    
    # DATA IMPORT 
    
    z = 8.397
    r = 24
    
    n_igm = np.load('../dataset/rho_z%.3f.npy' %z)  # density of intergalactic medium\n"
    n_src = np.load('../dataset/nsrc_z%.3f.npy' %z) # number of sources per volume\n"
    xi = np.load('../dataset/xHII_z%.3f.npy' %z) # ionization fraction"
    dims = n_igm.shape
    
    net = CNN.CNN()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    
    # TRAINING
    
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                
                n_igm_nbh = get_neighborhood(n_igm, x,y,z, r)
                n_src_nbh = get_neighborhood(n_src, x,y,z, r)
                loss_fn = nn.CrossEntropyLoss()
                optimizer.zero_grad()  # set the gradients to 0
                output = net(n_igm_nbh, n_src_nbh)
                loss = loss_fn(output, xi[x,y,z])  # compute loss function
                loss.backward()  # backpropagation
                optimizer.step()
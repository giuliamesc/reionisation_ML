import numpy as np
import torch


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

def preprocessing(n_igm,n_src, a=0, b=1):
    
    n_igm = 2.95 * 10**55 * n_igm
    
    n_min = n_igm.min()
    n_max = n_igm.max()
    n_igm = (n_igm - n_min) / (n_max - n_min) * (b-a) + a
    
    n_min = n_src.min()
    n_max = n_src.max()
    n_src = (n_src - n_min) / (n_max - n_min) * (b-a) + a
    
    return n_igm, n_src

z = 8.397
r = 24

np.random.seed(2021)

# DATA LOADING
    
n_igm = np.load('../dataset/rho_z%.3f.npy' %z)  # density of intergalactic medium
n_src = np.load('../dataset/nsrc_z%.3f.npy' %z) # number of sources per volume
xi = np.load('../dataset/xHII_z%.3f.npy' %z) #ionization rate

dims = n_igm.shape
D = dims[0]
S = 120 # number of sample points for the reduced database
n_valid = 300
N = S + n_valid

# PREPROCESSING

n_igm, n_src = preprocessing(n_igm, n_src)

# EXTRACTION OF THE 3000 INDEXES TO DEAL WITH A NOT TOO HUGE DATASET

ind1 = np.random.randint(0,D, N)
ind2 = np.random.randint(0,D, N)
ind3 = np.random.randint(0,D, N)

# STORAGE OF n_igm,n_src FOR THE NEIGHBORHOODS AND OF x_i FOR THE POINTS SELECTED

my_xi = torch.flatten(torch.Tensor(xi[ind1,ind2,ind3]))
my_xi = my_xi.numpy()
my_xi_valid = my_xi[S:] # last n_valid elements
my_xi = my_xi[:S] # first S elements
np.savetxt('cubes_small/xi_flatten.txt', my_xi)
np.savetxt('validation_small/xi_flatten.txt', my_xi_valid)

small_total = np.reshape(np.array([ind1,ind2,ind3]), [N,3])

for count in range(N):
    
    P = small_total[count,:]

    n_igm_nbh = torch.tensor(get_neighborhood(n_igm, P, r)).float()
    n_src_nbh = torch.tensor(get_neighborhood(n_src, P, r)).float()
    
    if (count < S) :
        np.save('cubes_small/n_igm_i%d.npy' % count, n_igm_nbh)
        np.save('cubes_small/n_src_i%d.npy' % count, n_src_nbh)
    else:
        np.save('validation_small/n_igm_i%d.npy' % (count - S), n_igm_nbh)
        np.save('validation_small/n_src_i%d.npy' % (count - S), n_src_nbh)
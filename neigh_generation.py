import parameters
import numpy as np
from shutil import rmtree
from os import makedirs

def preprocessing(n_igm,n_src):
    # this function preprocesses the features
    n_igm = 2.95 * 10**55 * n_igm # change of units of measure
    # mean, std deviation of the features
    mean_n_igm = np.mean(n_igm) 
    std_n_igm = np.std(n_igm)
    mean_n_src = np.mean(n_src)
    std_n_src = np.std(n_src)
    # normalization
    n_igm = (n_igm - mean_n_igm) / std_n_igm
    n_src = (n_src - mean_n_src) / std_n_src
    
    return n_igm, n_src

# reading the redshift and the radius from parameters.py
z = parameters.z 
r = parameters.r
np.random.seed(2021) # setting the seed for reproducibility

# DATA LOADING
path = './dataset/'
path_out = './cubes/'

# creation/cleaning of the folder "cubes"
makedirs(path_out, exist_ok=True)
rmtree(path_out) # to remove all previous generated cubes
makedirs(path_out) 

n_igm = np.load('%srho_z%.3f.npy' %(path, z))  # density of intergalactic medium
n_igm = n_igm / np.mean(n_igm) - 1. # typical transformation done in astrophysics
n_src = np.load('%snsrc_z%.3f.npy' %(path, z)) # number of sources per volume
xi = np.load('%sxHII_z%.3f.npy' %(path, z)) #ionization rate


D = n_igm.shape[0] # extracting the dimension of the space (D=3) GIUSTO????
S = parameters.S # reading the number of data we want to use from parameters.py


# generating three lists of S random indexes (at least far r from the border, so that the neighborhood is inside)

ind1 = np.random.randint(r+1, D-r, S)
ind2 = np.random.randint(r+1, D-r, S)
ind3 = np.random.randint(r+1, D-r, S)

# storage for the points selected of n_igm,n_src for the neighborhoods and of x_i 

target = []
cell_n_igm = []
cell_n_src = []
n_igm, n_src = preprocessing(n_igm,n_src) # preprocessing the data

for count in range(S):
    if (count%100==0):
        print (count, '/',S) # used for debugging
    i, j, k = ind1[count], ind2[count], ind3[count] # extracting the three coordinates

    target.append(xi[i,j,k]) # storing xi(i,j,k)
    cell_n_igm.append(n_igm[i,j,k]) # storing n_igm(i,j,k)
    cell_n_src.append(n_src[i,j,k]) # storing n_src(i,j,k)

    subvol_igm = n_igm[i-(r+1):i+r, j-(r+1):j+r, k-(r+1):k+r] # extraction of the neighborhood for n_igm
    subvol_src = n_src[i-(r+1):i+r, j-(r+1):j+r, k-(r+1):k+r] # extraction of the neighborhood for n_src
    
    norm_subvol_igm, norm_subvol_src = subvol_igm, subvol_src  # change of name since they are already normalized
    assert norm_subvol_igm.shape == (2*r+1,2*r+1,2*r+1) # COSA FA????
    assert norm_subvol_src.shape == (2*r+1,2*r+1,2*r+1) # COSA FA????
    
    # saving neighborhoods in .npy files, xi in .txt files
    np.save('%sn_igm_i%d.npy' %(path_out, count), norm_subvol_igm)
    np.save('%sn_src_i%d.npy' %(path_out, count), norm_subvol_src)
    np.savetxt('%sxi_flatten.txt' %path_out, target)
    np.savetxt('%sn_igm_flatten.txt' %path_out, cell_n_igm)
    np.savetxt('%sn_src_flatten.txt' %path_out, cell_n_src)

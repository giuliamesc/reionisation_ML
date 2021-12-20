import numpy as np

def preprocessing(n_igm,n_src, a=0, b=1):
    
    n_igm = 2.95 * 10**55 * n_igm
    
    n_min = n_igm.min()
    n_max = n_igm.max()
    n_igm = (n_igm - n_min) / (n_max - n_min) * (b-a) + a
    
    n_min = n_src.min()
    n_max = n_src.max()
    n_src = (n_src - n_min) / (n_max - n_min) * (b-a) + a
    
    return n_igm, n_src

##### IMPORTANT PARAMETERS TO SET #####
z = 8.397
r = 24
#######################################

np.random.seed(2021)

# DATA LOADING
path = './dataset/'
path_out = './cubes_CNN/'
n_igm = np.load('%srho_z%.3f.npy' %(path, z))  # density of intergalactic medium
n_igm = n_igm / np.mean(n_igm) - 1.
n_src = np.load('%snsrc_z%.3f.npy' %(path, z)) # number of sources per volume
xi = np.load('%sxHII_z%.3f.npy' %(path, z)) #ionization rate


D = n_igm.shape[0]
S = 10000    #10000 or 300 for the correlation plot


# PREPROCESSING

n_igm, n_src = preprocessing(n_igm, n_src)

# EXTRACTION OF THE S INDICES TO DEAL WITH A NOT TOO HUGE DATASET

ind1 = np.random.randint(0,D-r, S)
ind2 = np.random.randint(0,D-r, S)
ind3 = np.random.randint(0,D-r, S)

# STORAGE OF n_igm,n_src FOR THE NEIGHBORHOODS AND OF x_i FOR THE POINTS SELECTED

target = []
cell_n_igm = []
cell_n_src = []
n_igm = (n_igm - n_igm.min()) / (n_igm.max() - n_igm.min())
n_src = (n_src - n_src.min()) / (n_src.max() - n_src.min())
for count in range(S):
    if (count%100==0):
        print (count, '/',S)
    i, j, k = ind1[count], ind2[count], ind3[count]

    target.append(xi[i,j,k])
    cell_n_igm.append(n_igm[i,j,k])
    cell_n_src.append(n_src[i,j,k])

    subvol_igm = n_igm[i-(r+1):i+r, j-(r+1):j+r, k-(r+1):k+r]
    subvol_src = n_src[i-(r+1):i+r, j-(r+1):j+r, k-(r+1):k+r]
    
    norm_subvol_igm, norm_subvol_src = subvol_igm, subvol_src  #preprocessing(subvol_igm, subvol_src)
    assert norm_subvol_igm.shape == (2*r+1,2*r+1,2*r+1)
    assert norm_subvol_src.shape == (2*r+1,2*r+1,2*r+1)
    np.save('%sn_igm_i%d.npy' %(path_out, count), norm_subvol_igm)
    np.save('%sn_src_i%d.npy' %(path_out, count), norm_subvol_src)
    np.savetxt('%sxi_flatten.txt' %path_out, target)
    np.savetxt('%sn_igm_flatten.txt' %path_out, cell_n_igm)
    np.savetxt('%sn_src_flatten.txt' %path_out, cell_n_src)
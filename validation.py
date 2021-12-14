import numpy as np
import CNN
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import gc
import matplotlib.pyplot as plt


def plot_losses(epochs, loss_tr, loss_te):
    plt.plot(epochs, loss_tr, 'r') # training losses
    plt.plot(epochs, loss_te, 'g') # test losses
    plt.title('Losses trends')
    plt.show()


def correlation_plot(x_pred, x_true):
    plt.plot(x_true, x_true, 'r')  # y = x
    plt.plot(x_true, x_pred, 'bo')  # our actual prediction
    plt.show()
    #sigma = np.std(x_pred)
    #plt.plot(x_true, x_pred + sigma, 'r-')
    #plt.plot(x_true, x_pred - sigma, 'r-')



gc.collect()

# DATA IMPORT
# path to preprocessed dataset
path_preproc = 'validation/'
# path_preproc = 'validation/' # according to your choice of storage!
# number of data to use in the validation
dataset_size = 300

# load and prepare dataset with shape (dataset_size, input_type, channel_size, xdim, ydim, zdim)
X = np.zeros((dataset_size, 2, 1, 49, 49, 49))
for i in range(dataset_size):
    n_src = np.load('%sn_src_i%d.npy' % (path_preproc, i))
    n_igm = np.load('%sn_igm_i%d.npy' % (path_preproc, i))
    X[i, 0] = n_src[np.newaxis, ...]
    X[i, 1] = n_igm[np.newaxis, ...]
y = np.loadtxt('%sxi_flatten.txt' % path_preproc)[:dataset_size]



# convert numpy array to torch tensor
X_valid_src, X_valid_igm = torch.Tensor(X[:, 0, :, :, :, :]), torch.Tensor(X[:, 1, :, :, :, :])

del X
gc.collect()

y_valid = torch.Tensor(y)

del y
gc.collect()

# create pytorch dataset
valid_dataset = TensorDataset(X_valid_src, X_valid_igm, y_valid)

del X_valid_src
del X_valid_igm
gc.collect()

# create data loader to use in epoch for loop, for a batch of size 32
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

del valid_dataset
gc.collect()

print('Data loading successfully completed')


#Loading all the necessary
net = CNN.CNN()
PATH = '.\model\last_model.pt'
checkpoint = torch.load(PATH)
net.load_state_dict(checkpoint['model_state'])


#Commented code to plot also the losses
'''
train_losses = pickle.load(open(".\output_train", "rb"))
all_train_losses = train_losses["train_loss"]

test_losses = pickle.load(open(".\output_test","rb"))
all_test_losses = test_losses["test_loss"]
R2_test = pickle.load(open(".\R2_test_list","rb"))
all_R2_test = R2_test["R2_test"]
n_epochs = len(all_test_losses)
'''


#Code for the plots
predictions = []
validations = []
net.eval()  #It is necessary in order to do


for i in range(0,dataset_size):
        # Evaluate the network (forward pass)
        X_test_src, X_test_igm, y_test = next(iter(valid_loader))
        loss_fn = torch.nn.MSELoss()
        prediction = net(X_test_igm,X_test_src)
        predictions.append(prediction.detach().numpy())
        validations.append(y_test.detach().numpy())
        print("Iter :", i+1 ," of ", dataset_size)



#plot_losses(np.arange(1,n_epochs+1),all_train_losses,all_test_losses)
correlation_plot(predictions,validations)







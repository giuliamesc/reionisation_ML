import numpy as np
import CNN
import torch
from torch import nn
from torch import optim
import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import gc
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ignite
from ignite.contrib.metrics import regression as ign
import matplotlib.pyplot as plt



def print_train (loss, epoch, n_epochs, iter, n_iters, d_time, D_time, R2):
    time_ratio = float(n_iters - iter - 1)/((iter+1))
    remaining_time = D_time * time_ratio
    print ('epoch ', epoch+1,'/',n_epochs, ' |   iter ',iter+1, '/',n_iters, ' |  loss = ', format(torch.Tensor.detach(loss).item(), ".4f"), '|  R2 = ', format(R2, ".4f"), ' |   time: ', format(d_time, ".3f"), '  |   to end of epoch: ', format(remaining_time, ".3f"))


def print_test(loss, epoch, n_epochs, iter, n_iters, R2):
    print('epoch ', epoch+1,'/',n_epochs, ' |   iter ',iter+1, '/',n_iters, ' |  loss = ', format(torch.Tensor.detach(loss).item(), ".4f"), '|  R2 = ', format(R2, ".4f"))


def clock(curr_time, init_time): # to compute iteration time
    time_next = time.perf_counter()
    d_time = time_next - curr_time # delta-time between two iterations
    D_time = time_next - init_time # delta-time from the beginning
    return time_next, d_time, D_time

def plot_losses(epochs, loss_tr, loss_te):
    plt.plot(epochs, loss_tr, 'r') # training losses
    plt.plot(epochs, loss_te, 'g') # test losses
    plt.title('Losses trends')
    plt.show()
    
def correlation_plot(x_pred, x_true):
    plt.plot(x_true, x_true, 'r') # y = x
    plt.plot(x_true, x_pred, 'b') # our actual prediction
    sigma = np.std(x_pred)
    plt.plot(x_true, x_pred + sigma, 'r-')
    plt.plot(x_true, x_pred - sigma, 'r-')
    


# Main file

if __name__ == '__main__':

    gc.collect()

    # DATA IMPORT 
    # path to preprocessed dataset
    #path_preproc = '../cubes/'
    path_preproc = 'cubes/' # according to your choice of storage!
    # number of data to use in the training and validation
    dataset_size = 3000

    # load and prepare dataset with shape (dataset_size, input_type, channel_size, xdim, ydim, zdim)
    X = np.zeros((dataset_size, 2, 1, 49, 49, 49))
    for i in range(dataset_size):
        n_src = np.load('%sn_src_i%d.npy' % (path_preproc, i))
        n_igm = np.load('%sn_igm_i%d.npy' % (path_preproc, i))
        X[i, 0] = n_src[np.newaxis, ...]
        X[i, 1] = n_igm[np.newaxis, ...]
    y = np.loadtxt('%sxi_flatten.txt' % path_preproc)[:dataset_size]

    # split dataset into trianing (80%) and validation set (test_size = 20%)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2021)

    train_step = X_train.shape[0]//32 # // returns an approximation to integer of the division
    test_step = X_valid.shape[0]//32
    
    del X
    gc.collect()

    # convert numpy array to torch tensor
    X_train_src, X_train_igm = torch.Tensor(X_train[:, 0, :, :, :, :]), torch.Tensor(X_train[:, 1, :, :, :, :])
    X_valid_src, X_valid_igm = torch.Tensor(X_valid[:, 0, :, :, :, :]), torch.Tensor(X_valid[:, 1, :, :, :, :])

    del X_train
    del X_valid
    gc.collect()

    y_train = torch.Tensor(y_train)
    y_valid = torch.Tensor(y_valid)


    # create pytorch dataset
    train_dataset = TensorDataset(X_train_src, X_train_igm, y_train)
    valid_dataset = TensorDataset(X_valid_src, X_valid_igm, y_valid)

    # create data loader to use in epoch for loop, for a batch of size 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    del train_dataset
    del valid_dataset
    gc.collect()


    # CNN CREATION
    net = CNN.CNN()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)  #I suggest to try the performance with different learning rate : 0.1 , 1e-2 , 1e-3. However, remember that Adam is an adaptive method
    scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 7, min_lr = 1e-7)
    #PATH = '\model\last_model.txt'
    #net.load_state_dict(torch.load(PATH))

    # TRAINING
    #current_epoch = pickle.load(open(".\current_epoch","rb"))   #It is a python library
    #current_epoch = current_epoch["current_epoch"]
    epochs = 5
    #final_epoch = current_epoch + epochs
    pickle.dump({"current_epoch": epochs}, open(".\current_epoch", "wb"))


    #If you want to test a single batch (then also comment the inner for loop) and if you want to test a single data put batch_size = 1
    #X_train_src,X_train_igm,y_train = next(iter(train_loader))
    #X_test_src,X_test_igm,y_test = next(iter(valid_loader))
    #iter = 0


    prev_loss = 10**2 #  high initial value
    #prev_loss = pickle.load(open(".\min_loss","rb"))     # To load the min loss of the previous simulation
    #prev_loss = prev_loss["prev_loss"]

    all_losses = [] # will contain all the losses of the different epochs
    #all_losses = pickle.load(open(".\output", "rb"))  #To load the vector of losses
    #all_losses = all_losses["test_loss"]

    all_train_losses = [] # will contain all the losses of the different epochs
    #all_losses = pickle.load(open(".\output", "rb"))  #To load the vector of losses
    #all_losses = all_losses["test_loss"]



    for epoch in range(epochs):         #for epoch in range(current_epoch,final_epoch)

        print ('          TRAINING     epoch ',epoch+1,'/', epochs,'\n')

        init_time = time.perf_counter()
        curr_time = init_time
        loss_train = []
        net.train()   #Not fundamental, just to distinguish net.train() and net.eval() when we do validation
        for iter,(X_train_src,X_train_igm,y_train) in enumerate(train_loader):
            loss_fn = torch.nn.MSELoss()
            optimizer.zero_grad()  # set the gradients to 0
            output= net(X_train_igm, X_train_src) # forward
            loss = loss_fn(output, y_train)  # compute loss function
            loss_train.append(loss.item()) # storing the training losses
            R2 = ign.R2Score(output).compute() # add second input if we use GPU!
            loss.backward()  # backpropagation
            optimizer.step()

            curr_time, d_time, D_time = clock(curr_time, init_time)
            print_train(loss, epoch, epochs, iter, train_step, d_time, D_time, R2)    #the number of iterations should be training_set_size/batch_size ---> 3000*0.8/32
            
        loss_train = np.mean(loss_train)
        all_train_losses.append(loss_train) # storage of the training losses

        print('           TESTING     epoch ',epoch+1,'/', epochs,'\n')

        loss_test = []
        net.eval()  #It is necessary in order to do validation
        for iter,(X_test_src,X_test_igm,y_test) in enumerate(valid_loader):
            # Evaluate the network (forward pass)
            loss_fn = torch.nn.MSELoss()
            prediction = net(X_test_igm,X_test_src)
            R2 = ign.R2Score(prediction).compute() # add second input if we use GPU!
            loss = loss_fn(prediction,y_test)
            loss_test.append(loss.item())
            print_test(loss, epoch, epochs, iter, test_step, R2)

    
        # COMPARISONS AND SAVINGS

        loss_test = np.mean(loss_test)
        scheduler.step(loss_test)
        all_losses.append(loss_test)

        pickle.dump({"test_loss": all_losses}, open(".\output", "wb")) # it should overwrite the previous file
        print('\n Test loss of epoch ', epoch +1,' saved')

        if (loss_test < prev_loss):
            prev_loss = loss_test
            pickle.dump({"prev_loss" : prev_loss}, open(".\min_loss","wb"))  #We need also to save the minimum value of the test loss we achieved
            PATH = '\model\model_%d.txt' % epoch
            torch.save(net.state_dict(), PATH)
            print ('Model of epoch ', epoch+1,' saved')


    # Saving the last model used
    PATH = '\model\last_model.txt'
    torch.save(net.state_dict(), PATH)
    print('Last Model saved')


''' 
    #To reload the model 
    net = CNN.CNN()
    net.load_state_dict(torch.load(PATH))
'''



'''
    #Loading the test losses 
    data = pickle.load(open(".\output", "rb"))
    print(data["test_loss"])
'''











import numpy as np
import CNN
import FNN
import parameters
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import gc
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from shutil import rmtree
from os import makedirs




    
def correlation_plot(x_pred, x_true):
    # function generating a plot with the predicted values for x_i (y-axis) against the true ones (x-axis)
    x_pred, x_true = np.array(x_pred), np.array(x_true) # conversion into np.array
    plt.plot(x_true, x_true, 'r-') # plotting the line y = x
    plt.plot(x_true, x_true+0.68/2*x_true, 'k-', alpha=0.25) # plotting the line y = x+sigma*x
    plt.plot(x_true, x_true-0.68/2*x_true, 'k-', alpha=0.25) # plotting the line y = x-sigma*x
    plt.plot(x_true, x_pred, 'bo') # plotting our actual prediction
    plt.ylim(0,1)
    plt.ylabel('prediction')
    


# Main file

if __name__ == '__main__':
    
    
    ######### DATA LOADING #########

    gc.collect() # calling the garbage collector for lightening the memory usage
    
    # reading data from parameters.py
    dataset_size = parameters.S # dataset_size
    D = 2*parameters.r+1 # cube diameter
    epochs = parameters.epochs # number of epochs
    path_preproc = './cubes/' # path of the directory in which neighborhoods are stored


    # CNN case
    if (parameters.net_type == 'CNN'):
        X = np.zeros((dataset_size, 2, 1, D, D, D)) # initializing an empty array for data storage
        for i in range(dataset_size):
            n_src = np.load('%sn_src_i%d.npy' % (path_preproc, i), allow_pickle=True) # reading the file where n_src are stored
            n_igm = np.load('%sn_igm_i%d.npy' % (path_preproc, i), allow_pickle=True) # reading the file where n_igm are stored
            X[i, 0] = n_src[np.newaxis, ...] # filling with n_src the initialized empty array
            X[i, 1] = n_igm[np.newaxis, ...] # filling with n_igm the initialized empty array
        y = np.loadtxt('%sxi_flatten.txt' % path_preproc)[:dataset_size] # reading the file where xi are stored

        # split dataset into trianing and validation set 
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.01, random_state=2021)
        batch_size = parameters.batch_size # reading the batch_size from parameters.py
        train_step = X_train.shape[0]//batch_size # number of iterations of training; command // returns an approximation to integer of the division
        test_step = X_valid.shape[0]//batch_size # number of iterations of training;
    
        del X # cleaning X, that is now useless
        gc.collect() # calling the garbage collector

        # convert numpy array to torch tensor
        X_train_src, X_train_igm = torch.Tensor(X_train[:, 0, :, :, :, :]), torch.Tensor(X_train[:, 1, :, :, :, :])
        X_valid_src, X_valid_igm = torch.Tensor(X_valid[:, 0, :, :, :, :]), torch.Tensor(X_valid[:, 1, :, :, :, :])

        del X_train # cleaning X_train,X_valid and calling the garbage collector
        del X_valid
        gc.collect()

    # FNN case
    elif (parameters.net_type == 'FNN'):
        batch_size = 128 
        
        # loading stored values for n_src,n_igm,xi
        n_src = np.loadtxt('%sn_src_flatten.txt' % (path_preproc))[:dataset_size]
        n_igm = np.loadtxt('%sn_igm_flatten.txt' % (path_preproc))[:dataset_size]
        X = np.vstack((n_src, n_igm)).T[..., np.newaxis]
        y = np.loadtxt(path_preproc+'xi_flatten.txt')[:dataset_size]
        
        # split dataset into trianing and validation set
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=2021)
    
        # convert numpy array to torch tensor
        X_train_src, X_train_igm = torch.Tensor(X_train[:, 0, :]), torch.Tensor(X_train[:, 1, :])
        X_valid_src, X_valid_igm = torch.Tensor(X_valid[:, 0, :]), torch.Tensor(X_valid[:, 1, :])


    # convert numpy array to torch tensor
    y_train = torch.Tensor(y_train)
    y_valid = torch.Tensor(y_valid)

    # create pytorch dataset
    train_dataset = TensorDataset(X_train_src, X_train_igm, y_train)
    valid_dataset = TensorDataset(X_valid_src, X_valid_igm, y_valid)

    # create data loader to use in epoch for loop, for a batch of size batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    # cleaning some memory
    del train_dataset
    del valid_dataset
    gc.collect()
    
    print ('Data loading successfully completed')
    
    

    
    #------------------------------------------------------------------------------------
    
    
    
    ######## NET INITIALIZATION ######### 

    

    if(torch.cuda.is_available()):
        loss_fn = torch.nn.MSELoss().cuda() # for the case of laptop with local GPU
    else:
        loss_fn = torch.nn.MSELoss()
        
    # case of the first run
    if (parameters.first_run == True):
        
        # creation of the folder "checkpoints"
        makedirs('./checkpoints/', exist_ok=True)
        rmtree('./checkpoints/') # to remove all previous checkpoint files
        makedirs('./checkpoints/') 
        
        # loading the right NN class
        if (parameters.net_type=='CNN'):
            net = CNN.CNN()
        else:
            net = FNN.FNN()
            
        if(torch.cuda.is_available()): # for the case of laptop with local GPU
            net = net.cuda() 
            
        optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)  #optimizer Adam; adaptive learning rate method
        scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 10, min_lr=1e-12, verbose=True) # a scheduler to adjust the learning rate
        current_epoch = 0
        final_epoch = epochs
        prev_loss = 10**2 # high initial value
        all_test_losses = [] # will contain all the test losses of the different epochs
        all_train_losses = [] # will contain all the train losses of the different epochs
        all_R2_train = [] # will contain all the train R2s of the different epochs
        all_R2_test = [] # will contain all the test R2s of the different epochs
    else:
        # if it's not the first run, resume the training from last_model
        PATH = './checkpoints/last_model.pt'
        
        # loading the right NN class
        if (parameters.net_type=='CNN'):
            net = CNN.CNN()
        else:
            net = FNN.FNN()
        
        optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4) #optimizer Adam; adaptive learning rate method
        scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 7, min_lr = 1e-12, verbose=True) # a scheduler to adjust the learning rate
        
        #loading the information contained in the folder "checkpoints"
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['model_state'])
        if (torch.cuda.is_available()):  # for the case of laptop with local GPU
            net = net.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        current_epoch = checkpoint['epoch'] + 1   # since we save the last epoch done, we have to start from the correct one
        prev_loss = checkpoint['loss']
        final_epoch = current_epoch + epochs # updating the number of the final epoch

        train_losses = pickle.load(open("./checkpoints/loss_train.txt", "rb"))  # to load the vector of train losses
        test_losses = pickle.load(open("./checkpoints/loss_test.txt", "rb"))    # to load the vector of test losses
        all_test_losses = test_losses["test_loss"]
        all_train_losses = train_losses["train_loss"]
        
        #Loading R2 lists of train and test
        R2_train = pickle.load(open("./checkpoints/R2_train.txt","rb")) # to load the vector of train R2s
        R2_test = pickle.load(open("./checkpoints/R2_test.txt", "rb")) # to load the vector of test R2s
        all_R2_train = R2_train["R2_train"]
        all_R2_test = R2_test["R2_test"]
     
    
    print ('Net initialization successfully completed')

    #-------------------------------------------------------------------------------------------------------------------------


    for epoch in tqdm(range(current_epoch, final_epoch)):
        
        ##### TRAINING #####
        
        init_time = time.perf_counter() # to track the computational time
        curr_time = init_time
        loss_train = []
        R2_train = []
        net.train()   # not fundamental, just to distinguish net.train() and net.eval() when we do validation
        for iter,(X_train_src,X_train_igm,y_train) in enumerate(train_loader):
            if(torch.cuda.is_available()): # for a laptop with local GPU
                X_train_src,X_train_igm,y_train = X_train_src.cuda(), X_train_igm.cuda(), y_train.cuda()
            optimizer.zero_grad()  # set the gradients to 0
            output= net(X_train_igm, X_train_src) # forward pass
            loss = loss_fn(output, y_train)  # computing loss function
            loss_train.append(loss.item()) # storing the training losses
            R2 = r2_score(y_train.cpu().detach().numpy(), output.cpu().detach().numpy()) # computing R2 score
            R2_train.append(R2) # storing R2 score
            loss.backward()  # backpropagation
            optimizer.step() # optimizer step

            
        loss_train = np.mean(loss_train) # mean over all the iterations of the epoch
        all_train_losses.append(loss_train) # storage of the training losses

        R2_train = np.mean(R2_train) # mean over all the iterations of the epoch
        all_R2_train.append(R2_train) # storage of the training R2

        
        pickle.dump({"train_loss": all_train_losses}, open("./checkpoints/loss_train.txt", "wb")) # it overwrites the previous file
        pickle.dump({"R2_train": all_R2_train}, open("./checkpoints/R2_train.txt", "wb"))  # it overwrites the previous file

        ##### TEST #####

        loss_test = []
        R2_test = []
        net.eval()  # it is necessary in order to do validation
        for iter,(X_test_src,X_test_igm,y_test) in enumerate(valid_loader):
            if(torch.cuda.is_available()): # for a laptop with local GPU
                X_test_src, X_test_igm, y_test = X_test_src.cuda(), X_test_igm.cuda(), y_test.cuda()
                
            # evaluate the network (forward pass)
            prediction = net(X_test_igm,X_test_src)
            # computing and storing loss and R2 score
            R2 = r2_score(y_test.cpu().detach().numpy(),prediction.cpu().detach().numpy())
            R2_test.append(R2)
            loss = loss_fn(prediction,y_test)
            loss_test.append(loss.item())
            correlation_plot(prediction.cpu().detach().numpy(), y_test.cpu().detach().numpy()) # correlation plot


        plt.savefig('./checkpoints/corr_plot_%d.png' %(epoch+1), bbox_inches='tight') # saving correlation plot
        plt.clf() # to clear the current figure
    
        ## COMPARISONS AND SAVINGS ##

        loss_test = np.mean(loss_test) # mean over all the iterations
        scheduler.step(loss_test) # scheduler step
        all_test_losses.append(loss_test) # storage

        R2_test = np.mean(R2_test) # mean over all the iterations
        all_R2_test.append(R2_test) # storage

        pickle.dump({"test_loss": all_test_losses}, open("./checkpoints/loss_test.txt", "wb")) # it overwrites the previous file
        pickle.dump({"R2_test": all_R2_test}, open("./checkpoints/R2_test.txt", "wb"))  # it overwrites the previous file

        if (loss_test < prev_loss): # if our current model is better, update the best model saving the net state, loss value and R2 score
            prev_loss = loss_test
            PATH = './checkpoints/model_%d.pt' % epoch
            torch.save({'epoch': epoch,
                        'model_state': net.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'loss': prev_loss}, PATH)
        print('Epoch %d: loss=%.4f, val_loss=%.4f, R2=%.4f, val_R2=%.4f' %(epoch+1, loss_train, loss_test, R2_train, R2_test))


        # saving the last model used (to be sure, we save it each epoch)
        PATH = './checkpoints/last_model.pt'
        torch.save({'epoch': epoch,
                    'model_state': net.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'loss': prev_loss}, PATH)

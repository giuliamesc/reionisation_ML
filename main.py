import numpy as np
import CNN
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
import ignite
import ignite.contrib.metrics.regression as reg



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
    path_preproc = '../cubes/'
    #path_preproc = 'cubes/' # according to your choice of storage!
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
    
    
    
    print ('Data loading successfully completed')
    
    
    ###### IMPORTANT PARAMETERS TO SET #######
    epochs = 2
    first_run = False
    ##########################################
    
    
    
    # INITIALIZATION (depending on first run or not)
    
    if (first_run == True):
        net = CNN.CNN()
        optimizer = optim.Adam(net.parameters(), lr=1e-3)  #Adam is an adaptive learning rate method
        scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 7, min_lr = 1e-7)
        current_epoch = 0
        final_epoch = epochs
        prev_loss = 10**2 # high initial value
        all_test_losses = [] # will contain all the test losses of the different epochs
        all_train_losses = [] # will contain all the train losses of the different epochs
        all_R2_train = []
        all_R2_test = []
    else:
        #Resume the training
        PATH = '.\model\last_model.pt'
        net = CNN.CNN()
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 7, min_lr = 1e-7)
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        current_epoch = checkpoint['epoch']    #Because we save the last epoch as epoch + 1, so we can start from the correct one
        prev_loss = checkpoint['loss']
        final_epoch = current_epoch + epochs

        train_losses = pickle.load(open(".\output_train", "rb"))  #To load the vector of train losses
        test_losses = pickle.load(open(".\output_test", "rb"))    # To load the vector of test losses
        all_test_losses = test_losses["test_loss"]
        all_train_losses = train_losses["train_loss"]
        #Loading R2 lists of train and test
        R2_train = pickle.load(open(".\R2_train_list","rb"))
        R2_test = pickle.load(open(".\R2_test_list", "rb"))
        all_R2_train = R2_train["R2_train"]
        all_R2_test = R2_test["R2_test"]
     
    
    print ('Net initialization successfully completed')


    #If you want to test a single batch (then also comment the inner for loop) and if you want to test a single data put batch_size = 1
    #X_train_src,X_train_igm,y_train = next(iter(train_loader))
    #X_test_src,X_test_igm,y_test = next(iter(valid_loader))
    #iter = 0





    for epoch in range(current_epoch, final_epoch):

        print ('          TRAINING     epoch ',epoch+1,'/', final_epoch,'\n')

        init_time = time.perf_counter()
        curr_time = init_time
        loss_train = []
        R2_train = []
        net.train()   #Not fundamental, just to distinguish net.train() and net.eval() when we do validation
        for iter,(X_train_src,X_train_igm,y_train) in enumerate(train_loader):
            loss_fn = torch.nn.MSELoss()
            optimizer.zero_grad()  # set the gradients to 0
            output= net(X_train_igm, X_train_src) # forward
            loss = loss_fn(output, y_train)  # compute loss function
            loss_train.append(loss.item()) # storing the training losses
            R2 = r2_score(y_train.detach(),output.detach())
            R2_train.append(R2)
            loss.backward()  # backpropagation
            optimizer.step()

            curr_time, d_time, D_time = clock(curr_time, init_time)
            print_train(loss, epoch, final_epoch, iter, train_step, d_time, D_time, R2)    #the number of iterations should be training_set_size/batch_size ---> 3000*0.8/32
            
        loss_train = np.mean(loss_train)
        all_train_losses.append(loss_train) # storage of the training losses

        R2_train = np.mean(R2_train)
        all_R2_train.append(R2_train)

        
        pickle.dump({"train_loss": all_train_losses}, open(".\output_train", "wb")) # it overwrites the previous file
        print('\n')
        print('Train loss of epoch ', epoch + 1,' saved')

        pickle.dump({"R2_train": all_R2_train}, open(".\R2_train_list", "wb"))  # it overwrites the previous file
        print('R2 Train of epoch ', epoch + 1, ' saved')

        print('           TESTING     epoch ',epoch+1,'/', final_epoch,'\n')

        loss_test = []
        R2_test = []
        net.eval()  #It is necessary in order to do validation
        for iter,(X_test_src,X_test_igm,y_test) in enumerate(valid_loader):
            # Evaluate the network (forward pass)
            loss_fn = torch.nn.MSELoss()
            prediction = net(X_test_igm,X_test_src)
            R2 = r2_score(y_test.detach(),prediction.detach())
            R2_test.append(R2)
            loss = loss_fn(prediction,y_test)
            loss_test.append(loss.item())
            print_test(loss, epoch, final_epoch, iter, test_step, R2)

    
        # COMPARISONS AND SAVINGS

        loss_test = np.mean(loss_test)
        scheduler.step(loss_test)
        all_test_losses.append(loss_test)

        R2_test = np.mean(R2_test)
        all_R2_test.append(R2_test)

        pickle.dump({"test_loss": all_test_losses}, open(".\output_test", "wb")) # it overwrites the previous file
        print('\n')
        print('Test loss of epoch ', epoch + 1,' saved')

        pickle.dump({"R2_test": all_R2_test}, open(".\R2_test_list", "wb"))  # it overwrites the previous file
        print('R2 Test of epoch ', epoch + 1, ' saved')

        print("The test loss is: ", loss_test)
        print("We have to beat the loss: ", prev_loss)

        if (loss_test < prev_loss):
            prev_loss = loss_test
            PATH = '.\model\model_%d.pt' % epoch
            torch.save({'epoch': epoch + 1,
                        'model_state': net.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'loss': prev_loss}, PATH)
            print ('Model of epoch ', epoch+1,' saved')


        # Saving the last model used (to be sure, we save it each epoch)
        PATH = '.\model\last_model.pt'
        torch.save({'epoch': epoch + 1 ,
                    'model_state': net.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'loss': prev_loss}, PATH)
        print('Last model saved')







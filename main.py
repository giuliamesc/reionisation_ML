import numpy as np
import CNN
import torch
from torch import nn
from torch import optim
import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle





def printing (loss, epoch, n_epochs, iter, n_iters):
    print ('epoch ', epoch+1,'/',n_epochs, '   iter ',iter+1, '/',n_iters, '      loss = ', torch.Tensor.detach(loss).item())


def print_test(loss,iter,n_iters):
    print('iter ',iter+1, '/',n_iters, '      loss = ', torch.Tensor.detach(loss).item())
    


    
# Main file

if __name__ == '__main__':

    # DATA IMPORT 
    # path to preprocessed dataset
    path_preproc = 'cubes/'

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

    # convert numpy array to torch tensor
    X_train_src, X_train_igm = torch.Tensor(X_train[:, 0, :, :, :, :]), torch.Tensor(X_train[:, 1, :, :, :, :])
    X_valid_src, X_valid_igm = torch.Tensor(X_valid[:, 0, :, :, :, :]), torch.Tensor(X_valid[:, 1, :, :, :, :])

    y_train = torch.Tensor(y_train)
    y_valid = torch.Tensor(y_valid)


    # create pytorch dataset
    train_dataset = TensorDataset(X_train_src, X_train_igm, y_train)
    valid_dataset = TensorDataset(X_valid_src, X_valid_igm, y_valid)

    # create data loader to use in epoch for loop, for a batch of size 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    
    
    # CNN CREATION
    net = CNN.CNN()
    optimizer = optim.Adam(net.parameters(), lr=0.01)  #I suggest to try the performance with different learning rate : 0.1 , 1e-2 , 1e-3. However, remember that Adam is an adaptive method
    

    # TRAINING
    epochs = 5


    #If you want to test a single batch (then also comment the inner for loop) and if you want to test a single data put batch_size = 1
    #X_train_src,X_train_igm,y_train = next(iter(train_loader))
    #X_test_src,X_test_igm,y_test = next(iter(valid_loader))
    #iter = 1



    print("#############  TRAINING OF THE MODEL #############")
    for epoch in range(epochs):

        net.train()   #Not fundamental, just to distinguish net.train() and net.eval() when we do validation
        for iter,(X_train_src,X_train_igm,y_train) in enumerate(train_loader):
            loss_fn = torch.nn.MSELoss()
            optimizer.zero_grad()  # set the gradients to 0
            output= net(X_train_igm, X_train_src) # forward
            loss = loss_fn(output, y_train)  # compute loss function
            printing(loss, epoch, epochs, iter, 75)    #the number of iterations should be training_set_size/batch_size ---> 3000*0.8/32
            loss.backward()  # backpropagation
            optimizer.step()


    #VALIDATION
    print("############# VALIDATION OF OUR MODEL #############")
    loss_test = []
    net.eval()  #It is necessary in order to do validation
    for iter,(X_test_src,X_test_igm,y_test) in enumerate(valid_loader):
        # Evaluate the network (forward pass)
        loss_fn = torch.nn.MSELoss()
        prediction = net(X_test_igm,X_test_src)
        loss = loss_fn(prediction,y_test)
        loss_test.append(loss.item())
        print_test(loss,iter,600)

    #Saving the test losses
    pickle.dump({"test_loss": loss_test}, open(".\output", "wb"))
    print("Test Losses Saved")



    #Saving the model 
    PATH = 'model.txt'
    torch.save(net.state_dict(), PATH)
    print("Model saved")




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











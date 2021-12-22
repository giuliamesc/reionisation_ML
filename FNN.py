import torch
from torch import nn


# Construction of the Fully Connected Neural Network class

class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        
        
        # FULLY CONNECTED UPPER BRANCHES
        self.activation = nn.ELU() # choosing the Exponential Linear Unit activation function
        self.dropout = nn.Dropout(p=0.2) # setting the dropout probability to 0.2

        self.dense_1 = nn.Linear(in_features=1, out_features=15) # first dense layer
        self.dense_2 = nn.Linear(in_features=15, out_features=10) # second dense layer
        
        # FINAL FULLY CONNECTED BRANCH
        self.dense_8 = nn.Linear(in_features=10*2, out_features=1) # dense layer

        # FINAL ACTIVATION
        self.final_activation = nn.Sigmoid() # final activation function: Sigmoid 

    def forward(self, x1, x2):

        # RIGHT BRANCH
        out1 = self.dense_1(x1)
        out1 = self.activation(out1)
        out1 = self.dropout(out1)

        out1 = self.dense_2(out1)
        out1 = self.activation(out1)
        out1 = self.dropout(out1)

        # LEFT BRANCH
        out2 = self.dense_1(x2)
        out2 = self.activation(out2)
        out2 = self.dropout(out2)

        out2 = self.dense_2(out2)
        out2 = self.activation(out2)
        out2 = self.dropout(out2)

        # CENTRAL BRANCH
        out1 = torch.flatten(out1, 1) # flattening the output of the upper branches
        out2 = torch.flatten(out2, 1) # flattening the output of the upper branches
        out = torch.cat((out1, out2), dim = 1) # concatenating the tensors along the channel dim
        
        out = self.dense_8(out)
        #out = self.final_activation(out) # the output gets worse by adding it

        out = torch.flatten(out)   # in this way at the end we obtain a tensor of size torch.Size([batch_size]) instead of torch.Size([batch_size,1]), for a clean comparison with the y_train
        return out

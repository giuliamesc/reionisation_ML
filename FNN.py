import torch
from torch import nn


# Construction of the Convolutionary Neural Network class

class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        
        ### CANCELLARE ASSOLUTAMENTE
        # MAIN PARAMETERS
        self.kernel_size = 5
        self.padding = (self.kernel_size - 1)//2 
        self.stride_conv = 1
        
        self.kernel_pooling = 2
        self.stride_pool = self.kernel_pooling
        
        # FULLY CONNECTED UPPER BRANCHES
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=0.2)

        self.dense_1 = nn.Linear(in_features=1, out_features=15)
        self.dense_2 = nn.Linear(in_features=15, out_features=10)
        #self.dense_3 = nn.Linear(in_features=128, out_features=64)
        #self.dense_4 = nn.Linear(in_features=64, out_features=32)
        #self.dense_5 = nn.Linear(in_features=32, out_features=16)
        
        # FINAL FULLY CONNECTED BRANCH
        #self.dense_6 = nn.Linear(in_features=16*2, out_features=8)
        #self.dense_7 = nn.Linear(in_features=8, out_features=4)
        self.dense_8 = nn.Linear(in_features=10*2, out_features=1)

        # FINAL ACTIVATION
        self.final_activation = nn.Sigmoid()

    def forward(self, x1, x2):

        # RIGHT BRANCH
        out1 = self.dense_1(x1)
        out1 = self.activation(out1)
        out1 = self.dropout(out1)

        out1 = self.dense_2(out1)
        out1 = self.activation(out1)
        out1 = self.dropout(out1)
        """
        out1 = self.dense_3(out1)
        out1 = self.activation(out1)
        out1 = self.dropout(out1)

        out1 = self.dense_4(out1)
        out1 = self.dropout(out1)
        out1 = self.activation(out1)

        out1 = self.dense_5(out1)
        out1 = self.dropout(out1)
        out1 = self.activation(out1)
        """
        # LEFT BRANCH
        out2 = self.dense_1(x2)
        out2 = self.activation(out2)
        out2 = self.dropout(out2)

        out2 = self.dense_2(out2)
        out2 = self.activation(out2)
        out2 = self.dropout(out2)
        """
        out2 = self.dense_3(out2)
        out2 = self.dropout(out2)
        out2 = self.activation(out2)

        out2 = self.dense_4(out2)
        out2 = self.dropout(out2)
        out2 = self.activation(out2)

        out2 = self.dense_5(out2)
        out2 = self.dropout(out2)
        out2 = self.activation(out2)
        """
        # CENTRAL BRANCH
        out1 = torch.flatten(out1, 1)
        out2 = torch.flatten(out2, 1)
        out = torch.cat((out1, out2), dim = 1) # concatenating the tensors along the channel dim
        
        #out = self.dense_6(out)
        #out = self.dense_7(out)
        out = self.dense_8(out)
        #out = self.final_activation(out)

        out = torch.flatten(out)   #In this way at the end we obtain a tensor of size torch.Size([batch_size]) instead of torch.Size([batch_size,1]), for a clean comparison with the y_train
        return out

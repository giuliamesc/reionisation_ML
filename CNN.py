import torch
from torch import nn
import parameters


# Construction of the Convolutionary Neural Network class

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # MAIN PARAMETERS
        self.kernel_size = 5 # setting the kernel size for convolutional layers
        self.padding = (self.kernel_size - 1)//2 # setting valid padding 
        self.stride_conv = 1 # setting stride for convolutional layers
        
        self.kernel_pooling = 2 # setting the kernel size for pooling layers
        self.stride_pool = self.kernel_pooling
        self.in_feat_param = 8 * parameters.r**3 # right input dimension of the final fully connected central branch
                
        # CONVOLUTIONAL BRANCH
        self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=self.kernel_size, stride=self.stride_conv, padding=self.padding) # convolutional layer
        self.batch_normalization_1 = nn.BatchNorm3d(num_features=16) # bacth normalization
        
        self.conv3d_2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=self.stride_conv, padding=self.padding) # convolutional_layer
        self.batch_normalization_2 = nn.BatchNorm3d(num_features=32) # batch normalization
        
        self.activation = nn.LeakyReLU() # setting the L-ReLU activation function
        self.pooling_3d = nn.AvgPool3d(kernel_size=self.kernel_pooling) # pooling layer: average pooling
        
        # FULLY CONNECTED BRANCH
        self.dropout = nn.Dropout(p=0.1) # setting the dropout probability to 0.1
        
        self.dense_1 = nn.Linear(in_features=self.in_feat_param, out_features=256) # dense layer
        self.dense_2 = nn.Linear(in_features=256, out_features=128) # dense layer
        self.dense_3 = nn.Linear(in_features=128, out_features=64) # dense layer
        self.dense_4 = nn.Linear(in_features=64, out_features=32) # dense layer
        self.dense_5 = nn.Linear(in_features=32, out_features=16) # dense layer
        
        # FINAL PART OF THE FULLY CONNECTED BRANCH
        self.dense_6 = nn.Linear(in_features=16, out_features=8) # dense layer
        self.dense_7 = nn.Linear(in_features=8, out_features=4) # dense layer
        self.dense_8 = nn.Linear(in_features=4, out_features=1) # dense layer

        self.final_activation = nn.Sigmoid() # sigmoid as final activation function

    def forward(self, x1, x2):
        # LEFT BRANCH
        out1 = self.conv3d_1(x1)
        out1 = self.activation(out1)
        out1 = self.batch_normalization_1(out1)
        out1 = self.pooling_3d(out1)

        out1 = self.conv3d_2(out1)
        out1 = self.activation(out1)
        out1 = self.batch_normalization_2(out1)
        out1 = self.pooling_3d(out1)

        # RIGHT BRANCH
        out2 = self.conv3d_1(x2)
        out2 = self.activation(out2)
        out2 = self.batch_normalization_1(out2)
        out2 = self.pooling_3d(out2)

        out2 = self.conv3d_2(out2)
        out2 = self.activation(out2)
        out2 = self.batch_normalization_2(out2)
        out2 = self.pooling_3d(out2)

        # CENTRAL BRANCH
        out = torch.cat((out1, out2), dim = 1) # concatenating the tensors along the channel dim
        out = torch.flatten(out,1) # flattening the result of the concatenation

        out = self.dense_1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.dense_2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.dense_3(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.dense_4(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.dense_5(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.dense_6(out)
        out = self.dense_7(out)
        out = self.dense_8(out)


        out = torch.flatten(out)   # in this way at the end we obtain a tensor of size torch.Size([batch_size]) instead of torch.Size([batch_size,1]), for a clean comparison with the y_train

        
        return out

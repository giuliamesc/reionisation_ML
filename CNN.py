import torch
from torch import nn


# Construction of the Convolutionary Neural Network class

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # MAIN PARAMETERS
        self.kernel_size = 5
        self.padding = (self.kernel_size - 1)/2 
        self.stride_conv = 1
        
        self.kernel_pooling = 2
        self.stride_pool = self.kernel_pooling
        
        
        # LEFT BRANCH
        self.conv3d_6 = nn.Conv3d(in_channels=1, out_channels=128, kernel_size=self.kernel_size, stride=self.stride_conv, padding=2)
        self.batch_normalization_6 = nn.BatchNorm3d(num_features=128)
        self.leaky_re_lu_6 = nn.LeakyReLU()
        self.average_pooling_3d_6 = nn.AvgPool3d(kernel_size = self.kernel_pooling)
        
        self.conv3d_7 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, stride=self.stride_conv, padding=2)
        self.batch_normalization_7 = nn.BatchNorm3d(num_features=128)
        self.leaky_re_lu_7 = nn.LeakyReLU()
        self.average_pooling_3d_7 = nn.AvgPool3d(kernel_size = self.kernel_pooling)
        
        self.conv3d_8 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, stride=self.stride_conv, padding=2)
        self.batch_normalization_8 = nn.BatchNorm3d(num_features=128)
        self.leaky_re_lu_8 = nn.LeakyReLU()
        self.average_pooling_3d_8 = nn.AvgPool3d(kernel_size = self.kernel_pooling)
        
        
        # RIGHT BRANCH
        self.conv3d_15 = nn.Conv3d(in_channels=1, out_channels=128, kernel_size=self.kernel_size, stride=self.stride_conv, padding=2)
        self.batch_normalization_15 = nn.BatchNorm3d(num_features=128)
        self.leaky_re_lu_15 = nn.LeakyReLU()
        self.average_pooling_3d_15 = nn.AvgPool3d(kernel_size = self.kernel_pooling)
        
        self.conv3d_16 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, stride=self.stride_conv, padding=2)
        self.batch_normalization_16 = nn.BatchNorm3d(num_features=128)
        self.leaky_re_lu_16 = nn.LeakyReLU()
        self.average_pooling_3d_16 = nn.AvgPool3d(kernel_size = self.kernel_pooling)
        
        self.conv3d_17 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, stride=self.stride_conv, padding=2)
        self.batch_normalization_17 = nn.BatchNorm3d(num_features=128)
        self.leaky_re_lu_17 = nn.LeakyReLU()
        self.average_pooling_3d_17 = nn.AvgPool3d(kernel_size = self.kernel_pooling)
        
        # CENTRAL BRANCH
        
        #self.concatenate = torch.cat()
        #self.flatten = torch.flatten()
        self.dropout = nn.Dropout(p=0.8)

        self.dense = nn.Linear(in_features=55296, out_features=256)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.leaky_re_lu_18 = nn.LeakyReLU()
        
        self.dense_1 = nn.Linear(in_features=256, out_features=128)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.leaky_re_lu_19 = nn.LeakyReLU()
        
        self.dense_2 = nn.Linear(in_features=128, out_features=64)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.leaky_re_lu_20 = nn.LeakyReLU()

        self.dense_3 = nn.Linear(in_features=64, out_features=32)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.leaky_re_lu_21 = nn.LeakyReLU()
        
        self.dense_4 = nn.Linear(in_features=32, out_features=16)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.leaky_re_lu_22 = nn.LeakyReLU()
        
        self.dense_5 = nn.Linear(in_features=16, out_features=8)
        self.dense_6 = nn.Linear(in_features=8, out_features=4)
        self.dense_7 = nn.Linear(in_features=4, out_features=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x1, x2):
        
        # LEFT BRANCH
        
        out1 = self.conv3d_6(x1)
        out1 = self.batch_normalization_6(out1)
        out1 = self.leaky_re_lu_6(out1)
        out1 = self.average_pooling_3d_6(out1)

        out1 = self.conv3d_7(out1)
        out1 = self.batch_normalization_7(out1)
        out1 = self.leaky_re_lu_7(out1)
        out1 = self.average_pooling_3d_7(out1)

        out1 = self.conv3d_8(out1)
        out1 = self.batch_normalization_8(out1)
        out1 = self.leaky_re_lu_8(out1)
        out1 = self.average_pooling_3d_8(out1)
        
        # RIGHT BRANCH
        
        out2 = self.conv3d_15(x2)
        out2 = self.batch_normalization_15(out2)
        out2 = self.leaky_re_lu_15(out2)
        out2 = self.average_pooling_3d_15(out2)

        out2 = self.conv3d_16(out2)
        out2 = self.batch_normalization_16(out2)
        out2 = self.leaky_re_lu_16(out2)
        out2 = self.average_pooling_3d_16(out2)

        out2 = self.conv3d_17(out2)
        out2 = self.batch_normalization_17(out2)
        out2 = self.leaky_re_lu_17(out2)
        out2 = self.average_pooling_3d_17(out2)
        
        # CENTRAL BRANCH
        out = torch.cat((out1, out2), dim = 1) # concatenating the tensors along the channel dim
        out = torch.flatten(out,1)


        out = self.dropout(out)

        out = self.dense(out)
        out = self.dropout_1(out)
        out = self.leaky_re_lu_18(out)

        out = self.dense_1(out)
        out = self.dropout_2(out)
        out = self.leaky_re_lu_19(out)

        out = self.dense_2(out)
        out = self.dropout_3(out)
        out = self.leaky_re_lu_20(out)

        out = self.dense_3(out)
        out = self.dropout_4(out)
        out = self.leaky_re_lu_21(out)

        out = self.dense_4(out)
        out = self.dropout_5(out)
        out = self.leaky_re_lu_22(out)

        out = self.dense_5(out)
        out = self.dense_6(out)
        out = self.dense_7(out)
        out = self.final_activation(out)

        out = torch.flatten(out)   #In this way at the end we obtain a tensor of size torch.Size([batch_size]) instead of torch.Size([batch_size,1]), for a clean comparison with the y_train

        
        return out
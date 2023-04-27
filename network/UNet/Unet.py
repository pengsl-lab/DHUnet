# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import unetConv2, unetUp
from .init_weights import init_weights

class Unet(nn.Module):
    def __init__(self, in_channels=3, dims=[64, 128, 256, 512, 1024], num_classes=1000, is_deconv=True, is_batchnorm=True):
        super(Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        ############# encoder ###############
        # downsampling
        self.conv1 = unetConv2(self.in_channels, dims[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(dims[0], dims[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(dims[1], dims[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(dims[2], dims[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(dims[3], dims[4], self.is_batchnorm)

        ############# decoder ###############
        # upsampling
        self.up_concat4 = unetUp(dims[4], dims[3], self.is_deconv)
        self.up_concat3 = unetUp(dims[3], dims[2], self.is_deconv)
        self.up_concat2 = unetUp(dims[2], dims[1], self.is_deconv)
        self.up_concat1 = unetUp(dims[1], dims[0], self.is_deconv)

        self.outconv1 = nn.Conv2d(dims[0], num_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)  
        maxpool1 = self.maxpool1(conv1)  
        
        conv2 = self.conv2(maxpool1) 
        maxpool2 = self.maxpool2(conv2)  

        conv3 = self.conv3(maxpool2)  
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)  
        maxpool4 = self.maxpool4(conv4)  

        center = self.center(maxpool4)  

        up4 = self.up_concat4(center, conv4) 
        up3 = self.up_concat3(up4, conv3) 
        up2 = self.up_concat2(up3, conv2)  
        up1 = self.up_concat1(up2, conv1)

        d1 = self.outconv1(up1) 

        return d1

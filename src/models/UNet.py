from typing import List

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


#double conv for each layer
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(
            self, in_channels : int = 3, out_channels : int = 1, features : List[int] = [64, 128, 256, 512]
    ):
        super(UNet, self).__init__()
        #create list to store layers
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList() 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #reduces spatial dimensions

        #down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) #in channels -> 3 initially then we turn it into 64
            in_channels = feature #set in_channels to feature so we can go from 64 -> 128 etc
        
        #Up part of Unet
        for feature in reversed(features):
            self.ups.append( #skip connection input is 2x adjacent layer output
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2) #double height and width of image (input will be 2x output (downsampling))
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) #512 -? 1024
        self.finalConv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = [] #store skip connections
        for down in self.downs: #goes through down blocks
            x = down(x) #2 conv in block
            skip_connections.append(x)#store skip for later
            x = self.pool(x)
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        
        for i in range(0, len(self.ups), 2): #up -> doubleconv
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2] #use same index, divide by 2 to keep the steps linear 1, 2, 3, 4, etc
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim = 1) #add skip connection along dim 1 I think width
            x = self.ups[i+1](concat_skip)

        return self.finalConv(x)
    
    

def test():
    x = torch.rand((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1)
    logit = model(x)
    print(logit.shape)
    print(x.shape)
    assert logit.shape == x.shape

if __name__ == '__main__':
    test()
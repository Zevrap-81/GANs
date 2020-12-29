import torch
import torch.nn as nn


class DC_Discriminator(nn.Module):
    def __init__(self, in_channels, ndf, m):
        super().__init__()
        self.in_channels= in_channels
        self.m=m
        self.model_layers=[]
          
        for i in self.m[:-1]:
            self.model_layers.append(self.getD_Layer(in_channels, ndf*i))
            in_channels= ndf*i
        self.model_layers.append(self.getD_Layer(in_channels, 1, final_layer= True))
    
        self.model= nn.Sequential(*self.model_layers)
        
        
    def getD_Layer(self,in_channels, out_channels, final_layer= False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1 ), 
                nn.BatchNorm2d(out_channels),  
                nn.LeakyReLU(0.2))
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0 ), 
                nn.Sigmoid()
                )
    def forward(self, img):
        return self.model(img)



class DC_Generator(nn.Module):
    def __init__(self, r_size, in_channels, ngf, m):
        super().__init__()
        self.r_size= r_size
        self.model_layers=[]
        self.in_channels= in_channels
        
        # the multiplier for the out_channels in each layer
        self.m=m

        #creating the list of layer networks
        in_channels= r_size
        for index, i in enumerate(self.m[:0:-1]):
            if index==0:
                self.model_layers.append(self.getD_Layer(in_channels, ngf*i, first_layer= True))
                in_channels= ngf*i
            else:
                self.model_layers.append(self.getD_Layer(in_channels, ngf*i))
                in_channels= ngf*i
        self.model_layers.append(self.getD_Layer(in_channels, self.in_channels, final_layer= True))
        
        self.model= nn.Sequential(*self.model_layers)
    
    def getD_Layer(self, in_channels, out_channels, first_layer=False, final_layer= False):
        if first_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0 ), 
                nn.BatchNorm2d(out_channels),  
                nn.ReLU())
        elif not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1 ), 
                nn.BatchNorm2d(out_channels),  
                nn.ReLU())
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1 ), 
                nn.Tanh()
                )
    def forward(self, rand_vect):
        return self.model(rand_vect.view(-1, self.r_size, 1, 1))   
        

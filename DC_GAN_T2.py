import torch
import torch.nn as nn
#size of feature maps
ndf= 28 #of discriminator
ngf= 28 #of generator

#number of channels in dataset
in_channels= 1

# the multiplier for the out_channels in each layer
m=[2**i for i in range(math.floor(math.log2(ndf))-1)]

class DC_Gen_t2(nn.Module):
    def __init__(self, r_size, in_channels, ngf, m):
        super().__init__()
        self.r_size= r_size
        self.model_layers=[]
        self.in_channels= in_channels
        
        # the multiplier for the out_channels in each layer
        self.m=m

        #creating the list of layer networks
        in_channels= r_size
        for index, i in enumerate(self.m[-2::-1]):
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
                #input batch_size x r_size x 1 x 1
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False ), 
                # output batch_size x out_channel(56) x 4 x 4
                nn.BatchNorm2d(out_channels),  
                nn.ReLU())
        elif not final_layer:
            return nn.Sequential(
                # input batch_size x out_channel(56) x 4 x 4
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=3, padding=1, bias=False ),
                # output batch_size x out_channel(28) x 10 x 10
                nn.BatchNorm2d(out_channels),  
                nn.ReLU())
        else:
            return nn.Sequential(
                # input batch_size x out_channel(28) x 10 x 10
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=3, padding=1, bias=False ), 
                # output batch_size x out_channel(1) x 28 x 28
                nn.Tanh()
                )
    def forward(self, rand_vect):
        rand_vect= rand_vect.view(-1, self.r_size, 1, 1)
        return self.model(rand_vect)   
        
        
class DC_Disc_t2(nn.Module):
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
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=3, padding=1, bias=False), 
                nn.BatchNorm2d(out_channels),  
                nn.LeakyReLU(0.2))
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=3, padding=0, bias=False ), 
                #out_put batch_size x in_channels x 1 x 1
                nn.Sigmoid()
                )
    def forward(self, img):
        return self.model(img)
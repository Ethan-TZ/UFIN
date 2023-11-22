from turtle import forward
import torch
from torch import Tensor, nn
from Utils import Config

class mLinear(nn.Module):
    def __init__(self, inshape, outshape, num_scenes, dp_rate = 0.1, drop = False) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_scenes, inshape, outshape))
        self.b = nn.Parameter(torch.zeros(num_scenes, outshape))
        self.drop = drop
        self.drop_out = nn.Dropout(p = dp_rate)
        nn.init.xavier_normal_(self.W)
    
    def forward(self, input):
        x, scene = input
        w, b = self.W[scene.long()], self.b[scene.long()] #[b,]
        x =  (x[:,None,:]@w).squeeze(1) + b
        if not self.drop:
            x = torch.relu(x)
        x = self.drop_out(x)
        return x, scene

class mDNN(nn.Module):
    def __init__(self , config:Config , Shape , drop_last = True , act = None , autoMatch = True, dp_rate = 0.1):
        super().__init__()
        layers = []
        self.autoMatch = autoMatch
        if self.autoMatch:
            Shape[0] = (len(config.feature_stastic) -1 ) * config.embedding_size
        for i in range(0 , len(Shape) - 2):
            hidden = mLinear(Shape[i] , Shape[i + 1] , 7, dp_rate, False)
            layers.append(hidden)
        layers.append(mLinear(Shape[-2] , Shape[-1] , 7, dp_rate, drop= True))
        self.net = nn.Sequential(*layers)
    
    def forward(self , input : Tensor):
        x, scene = input
        if self.autoMatch:
            x = x.reshape(x.shape[0] , -1)
        return self.net((x, scene))

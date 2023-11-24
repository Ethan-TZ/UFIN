import torch
from torch import nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
from Model.Layers.mDNN  import mDNN
 # res or  stack mlp

class SharedBottom(BasicModel):
    def __init__(self, config : Config):
        super().__init__(config)
        self.top_mlp = mDNN(config, [0, 256, 256, 256, 1], dp_rate = 0.1)
        

    def FeatureInteraction(self , feature , sparse_input, *kwards):
        x, scene = feature, sparse_input['scene'] - 1
        x, scene = self.top_mlp([x, scene])
        self.logits = x
        self.output = torch.sigmoid(self.logits)
        return self.output
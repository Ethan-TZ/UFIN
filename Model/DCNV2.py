import torch
from torch import nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN

class CrossNetComp(nn.Module):
    def __init__(self, config:Config):
        super(CrossNetComp, self).__init__()
        hiddenSize = (len(config.feature_stastic) - 1) * config.embedding_size
        self.W = nn.Linear(hiddenSize , hiddenSize)
        nn.init.normal_(self.W.weight, mean = 0, std = 0.01)
        
    def forward(self, base, cross):
        result = base * self.W(cross) + cross
        return result

class CrossNet(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        self.depth = config.depth
        self.crossnet = nn.ModuleList([CrossNetComp(config) for i in range(self.depth)])
        self.linear = nn.Linear((len(config.feature_stastic) - 1) * config.embedding_size , 1)
        nn.init.normal_(self.linear.weight)
    
    def FeatureInteraction(self , feature , sparse_input, *kargs):
        feature = feature.reshape(feature.shape[0] , -1)
        base = feature
        cross = feature
        for i in range(self.depth):
            cross = self.crossnet[i](base , cross)
        self.logits = self.linear(cross)
        self.output = torch.sigmoid(self.logits)
        return self.output

class DCNV2(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        self.depth = config.depth
        self.mlplist = config.mlp
        self.dnn = DNN(config , self.mlplist)
        self.crossnet = nn.ModuleList([CrossNetComp(config) for i in range(self.depth)])
        self.linear = nn.Linear((len(config.feature_stastic) - 1) * config.embedding_size , 1)
        self.fine_tune_adaptiver = nn.Linear(96, 1, bias = False)
        field_num = self.field_num = len(config.feature_stastic) - 1
        self.C = nn.Parameter(torch.ones(1, field_num, 1))

    def FeatureInteraction(self , feature , sparse_input, *kargs):
        def ff(feature):
            dnn = self.dnn(feature)
            feature = feature.reshape(feature.shape[0] , -1)
            base = feature
            cross = feature
            for i in range(self.depth):
                cross = self.crossnet[i](base , cross)
            
            logits = self.linear(cross) + dnn
            return logits
        
        self.logits = ff(feature)
        self.output = torch.sigmoid(self.logits)
        return self.output
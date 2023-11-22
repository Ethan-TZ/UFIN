from abc import abstractmethod
from cmath import log
from curses.ascii import EM
from turtle import forward
import torch
from torch import embedding, nn
from Model.Layers.Embedding import Embedding, i_Embedding
from abc import abstractmethod

class BasicModel(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.embedding_layer = Embedding(config)
        self.i_embedding_layer = i_Embedding(config)
        self.g_embedding_layer = Embedding(config)
        self.backbone = []

    def forward(self , sparse_input, dense_input = None):
        self.dense_input = self.embedding_layer(sparse_input)
        predict = self.FeatureInteraction(self.dense_input , sparse_input, self.i_embedding_layer(sparse_input, "no_id"), self.g_embedding_layer(sparse_input))
        return predict
    
    @abstractmethod
    def FeatureInteraction(self , dense_input , sparse_input, *kwrds):
        pass

    def RegularLoss(self , weight):
        if weight == 0:
            return 0
        loss = 0
        for _ in self.backbone:
            comp = getattr(self , _)
            if isinstance(comp , nn.Parameter):
                loss += torch.norm(comp , p = 2)
                continue
            for params in comp.parameters():
                loss += torch.norm(params , p = 2)
        return loss * weight
    
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
#        self.deep_r_embedding_layer = Embedding(config)
#        self.deep_i_embedding_layer = Embedding(config)
        self.backbone = []
        if hasattr(config , 'n_ensembler'):
            self.w = nn.Parameter(torch.ones(config.n_ensembler))
            self.b = nn.Parameter(torch.zeros(config.n_ensembler))
            self.tau = config.tau

    def forward(self , sparse_input, dense_input = None):
        self.dense_input = self.embedding_layer(sparse_input)
        predict = self.FeatureInteraction(self.dense_input , sparse_input, self.i_embedding_layer(sparse_input, "no_id"), self.g_embedding_layer(sparse_input))#, self.deep_i_embedding_layer(sparse_input), self.deep_r_embedding_layer(sparse_input))
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
    
    def get_aux_loss(self, logits_of_teacher):
        pass
        alpha = torch.softmax(self.w[None,:] * logits_of_teacher + self.b[None,:] , dim=-1)
        y_T = torch.sum(torch.sigmoid(logits_of_teacher / self.tau) * alpha , dim=-1)
        y_S = (torch.sigmoid(self.logits / self.tau)).squeeze(-1)
        return torch.mean(y_T * torch.log(y_S + 1e-6) + (1-y_T) * torch.log(1 - y_S + 1e-6))
    
    def get_params(self):
         total = sum([param.nelement() for param in self.parameters()])
         embedding = sum([param.nelement() for param in self.embedding_layer.parameters()])
         i_embedding = sum([param.nelement() for param in self.i_embedding_layer.parameters()])
         return total - embedding - i_embedding
    
    def latency(self,num_fields, embedding_size):
        from tqdm import tqdm
        for i in tqdm(range(10000)):
            xx = torch.randn(1024,num_fields, embedding_size)
            self.FeatureInteraction(xx, xx, xx)
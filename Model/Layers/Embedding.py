import torch
from torch import nn
from Utils import Config
from Model.Layers.DNN import DNN
import numpy as np
import random as r

class Embedding(nn.Module):
    def __init__(self , config : Config):
        super().__init__()
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        self.field = []
        for feature , numb in config.feature_stastic.items():
            if feature != 'label':
                self.field += [feature]
                self.embedding[feature] = nn.Embedding(numb + 1 , 16)
        for _, value in self.embedding.items():
            nn.init.xavier_uniform_(value.weight)

    def forward(self , data, mode = "normal"):
        batch = len(data['label'])
        out = []
        if mode == "normal":
            for name in self.field:
                if name != 'label' and name in self.embedding:
                        raw = data[name]
                        raw = torch.Tensor(raw).long().cuda()
                        phase = self.embedding[name](raw)
                        if name in ['category', 'title', 'description']:
                            phase = torch.mean(phase * (raw != 0)[:,:,None], dim = 1, keepdim=True)
                        else:
                            phase = phase[:,None,:]
                        out.append(phase)
        elif mode == "no_id":
            for name in self.field:
                if name != 'label' and name in self.embedding:
                        raw = data[name]
                        phase = self.embedding[name](torch.LongTensor(raw).cuda())[:,None,:]
                        if 'id' in name:
                            phase = phase.data
                        out.append(phase)
        else:
            for name , raw in data.items():
                if name != 'label':
                    if name == self.user_field or name == self.item_field :#or name in self.context_field:
                        out.append( torch.repeat_interleave(torch.sum(self.embedding[name].weight * (self.pop[name][:,None]) , dim = 0)[None,None,:], batch,0) )
                    else:
                        out.append(self.embedding[name](raw.long().cuda())[:,None,:])
        return torch.cat(out , dim = -2)


class i_Embedding(nn.Module):
    def __init__(self , config : Config):
        super().__init__()
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        self.modulus: nn.ParameterDict[str:nn.Embedding] = nn.ParameterDict()
        self.field = []
        
        for feature , numb in config.feature_stastic.items():
            if feature != 'label' and feature in ['user_id']:
                self.field += [feature]
                self.embedding[feature] = nn.Embedding(numb + 1 ,  256 if feature != 'user_id' else 16)
                self.modulus[feature] = nn.Parameter(torch.ones(1).float())

        for _, value in self.embedding.items():
            nn.init.xavier_uniform_(value.weight)

    def get_popularity(self , train_data):
        #return
        for batch in train_data:
            for j in batch[self.user_field]:
                self.pop[self.user_field][j] += 1
            for j in batch[self.item_field]:
                self.pop[self.item_field][j] += 1            
        self.pop[self.user_field] /= torch.sum(self.pop[self.user_field])
        self.pop[self.item_field] /= torch.sum(self.pop[self.item_field])

    def forward(self , data, mode = "normal"):
        batch = len(data['label'])
        u_out, i_out = [], []
        out = []
        if mode == "normal":
            for name in self.field:
                if name != 'label' and name in self.embedding:
                        raw = data[name].long()
                        phase = self.embedding[name]((raw).cuda())[:,None,:]
                        out.append(phase)
        elif mode == "no_id":
            user_side = ['user_id']
            item_side = []
            for name in self.field:
                if name != 'label' and name in user_side + item_side:
                        raw = data[name]
                        if not isinstance(raw, torch.Tensor):
                            raw = torch.Tensor(raw)
                        
                        raw = raw.cuda()
                        phase = self.embedding[name]((raw).long())
                        if name in ['category', 'title', 'description']:
                            phase = torch.mean(phase * (raw != 0)[:,:,None], dim = 1, keepdim=True)
                        else:
                            phase = phase[:,None,:]

                        if name in user_side:
                            u_out.append(phase)
                        else:
                            i_out.append(phase)

        else:
            for name , raw in data.items():
                if name != 'label':
                    if name == self.user_field or name == self.item_field :#or name in self.context_field:
                        out.append( torch.repeat_interleave(torch.sum(self.embedding[name].weight * (self.pop[name][:,None]) , dim = 0)[None,None,:], batch,0) )
                    else:
                        out.append(self.embedding[name](raw.long().cuda())[:,None,:])

        return torch.cat(u_out, dim = -2)
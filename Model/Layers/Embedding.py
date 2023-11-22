import torch
from torch import nn
from Utils import Config
from Model.Layers.DNN import DNN
import numpy as np
import random as r

class AttentionLayer(nn.Module):
    def __init__(self , headNum = 2 , att_emb = 8 , input_emb = 16):
        super().__init__()
        self.headNum = headNum
        self.att_emb = att_emb
        self.input_emb = input_emb
        self.Query = nn.Parameter(torch.zeros(1 , self.headNum , 1 , self.input_emb , self.att_emb))
        self.Key = nn.Parameter(torch.zeros(1 , self.headNum , 1 , self.input_emb , self.att_emb))
        self.Value = nn.Parameter(torch.zeros(1 , self.headNum , 1 , self.input_emb , self.att_emb))
        self.res = nn.Parameter(torch.zeros(self.input_emb , self.headNum * self.att_emb))
        self.init()

    def forward(self, feature):
        #[b , 1 , f , d ]
        res = feature @ self.res
        feature = feature.view(feature.shape[0] , 1 , feature.shape[1] , 1 , -1 )
        query = (feature @ self.Query).squeeze(3)
        key = (feature @ self.Key).squeeze(3)
        value = (feature @ self.Value).squeeze(3)

        # [b , h , f , f]
        score = torch.softmax(query @ key.transpose(-1,-2) , dim = -1)
        # [b , h , f , d]
        em = score @ value
        em = torch.transpose(em , 1  , 2)
        em = em.reshape(res.shape[0],res.shape[1],res.shape[2])
        
        return torch.relu(em + res)
    
    def init(self):
        for params in self.parameters():
            nn.init.xavier_uniform_(params , gain=1.414)


class Embedding(nn.Module):
    def __init__(self , config : Config):
        super().__init__()
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        self.modulus: nn.ParameterDict[str:nn.Embedding] = nn.ParameterDict()
        self.P: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        self.Q: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()

        self.field = []
        for feature , numb in config.feature_stastic.items():
            if feature != 'label':
                self.field += [feature]
                self.embedding[feature] = nn.Embedding(numb + 1 , 16)
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
                            
                        out.append(phase)#+ (self.embedding[name+'meta'].weight[None,:] if name + 'meta' in self.embedding else 0))
        else:
            for name , raw in data.items():
                if name != 'label':
                    if name == self.user_field or name == self.item_field :#or name in self.context_field:
                        #out.append(torch.repeat_interleave(self.embedding[name+'meta'].weight[None,:], batch,0))
                        out.append( torch.repeat_interleave(torch.sum(self.embedding[name].weight * (self.pop[name][:,None]) , dim = 0)[None,None,:], batch,0) )
                    else:
                        out.append(self.embedding[name](raw.long().cuda())[:,None,:])
        return torch.cat(out , dim = -2)


class i_Embedding(nn.Module):
    def __init__(self , config : Config):
        super().__init__()
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        self.modulus: nn.ParameterDict[str:nn.Embedding] = nn.ParameterDict()
        self.P: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        self.Q: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        # self.user_field = config.user_field
        # self.item_field = config.item_field
        # self.pop = {}
        self.extracter = AttentionLayer()
        # from ..PLM import PLM
        # self.plm = PLM(config)

        self.field = []
        # config.feature_stastic['genre'] = 10000
        # config.feature_stastic['movie_title'] = 10000
        
        self.cls_embedding = torch.nn.Parameter(torch.randn(1, 1, 16) * 0.01)
        # self.cls_embedding_user = torch.nn.Parameter(torch.randn(1, 1, 16) * 0.01)
        # config.feature_stastic['weekday'] = 10
        # config.feature_stastic['hour'] = 100
        for feature , numb in config.feature_stastic.items():
            if feature != 'label' and feature in ['user_id']:
                self.field += [feature]
                self.embedding[feature] = nn.Embedding(numb + 1 ,  256 if feature != 'user_id' else 16)
                self.modulus[feature] = nn.Parameter(torch.ones(1).float())
                self.P[feature] = DNN(config, [16, 64, 64, 1], autoMatch= False, drop_last= False)
                self.Q[feature] = nn.Linear(config.embedding_size, config.embedding_size)
                # if feature == self.user_field or feature == self.item_field:
                #     self.pop[feature] = torch.zeros(numb + 1).cuda()
                #     self.embedding[feature+'meta'] = nn.Embedding(1 , config.embedding_size)
                # else:
                #     self.embedding[feature+'meta'] = nn.Embedding(numb + 1 , config.embedding_size)

        # for feature , numb in config.feature_stastic.items():
        #     if feature != 'label' and feature in ['item_id']:
        #         self.field += [feature]
        #         self.embedding[feature] = nn.Embedding(numb + 1 , 64)
        #         self.modulus[feature] = nn.Parameter(torch.ones(1).float())
        #         self.P[feature] = DNN(config, [16, 64, 64, 1], autoMatch= False, drop_last= False)
        #         self.Q[feature] = nn.Linear(config.embedding_size, config.embedding_size)
        #         # if feature == self.user_field or feature == self.item_field:
        #         #     self.pop[feature] = torch.zeros(numb + 1).cuda()
        #         #     self.embedding[feature+'meta'] = nn.Embedding(1 , config.embedding_size)
        #         # else:
        #         #     self.embedding[feature+'meta'] = nn.Embedding(numb + 1 , config.embedding_size)

        for _, value in self.embedding.items():
            nn.init.xavier_uniform_(value.weight)
            #nn.init.uniform_(value.weight, a = -3.14 / 3, b = 3.14 /)

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
                        # if 'id' in name and r.random() < 1.1:
                        #     phase = torch.zeros_like(phase).cuda()
                        # modulus = self.modulus[name]
                        #phase = self.P['user_id'](phase) * torch.sin(phase)
                        out.append(phase)#+ (self.embedding[name+'meta'].weight[None,:] if name + 'meta' in self.embedding else 0))
        elif mode == "no_id":
            user_side = ['user_id']
            item_side = []
            # user_side = ['age', 'zip_code', 'occupation', 'gender', 'release_year']
            # item_side = ['item_id', 'user_id']
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

                            #phase = torch.mean(phase, dim = 1, keepdim= True)
                        # elif name == 'gender':
                        #     res = []
                        #     for field in user_side:
                        #         rr = data[field]   
                        #         dd = self.embedding[field](torch.LongTensor(rr).cuda())[:,None,:]
                        #         res.append(dd)
                        #     xx = torch.repeat_interleave(self.cls_embedding, res[0].shape[0], 0)
                        #     res.append(xx)
                        #     phase = torch.cat(res, dim = 1)
                        #     phase = self.extracter(phase)[:,0,:].unsqueeze(1)
                        if name in user_side:
                            u_out.append(phase)
                        else:
                            i_out.append(phase)
                        # if name != 'movie_title':
                        #     phase = self.embedding[name](torch.LongTensor(raw).cuda())[:,None,:]
                        #out.append(phase)#+ (self.embedding[name+'meta'].weight[None,:] if name + 'meta' in self.embedding else 0))
        else:
            for name , raw in data.items():
                if name != 'label':
                    if name == self.user_field or name == self.item_field :#or name in self.context_field:
                        #out.append(torch.repeat_interleave(self.embedding[name+'meta'].weight[None,:], batch,0))
                        out.append( torch.repeat_interleave(torch.sum(self.embedding[name].weight * (self.pop[name][:,None]) , dim = 0)[None,None,:], batch,0) )
                    else:
                        out.append(self.embedding[name](raw.long().cuda())[:,None,:])
        # i_out = torch.cat(i_out, dim = 1)
        # i_out = torch.sum(i_out, dim = 1, keepdim=True)
        return torch.cat(u_out, dim = -2)
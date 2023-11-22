from turtle import forward
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
import torch
from torch import nn


# 一份进block, 一份进deep，然后合并
class Bridge_Block(nn.Module):
    def __init__(self, inshape, outshape, act = None) -> None:
        super().__init__()
        hidden_size = 100

        self.c_dnn = nn.Linear(inshape, outshape)
        self.r_dnn = nn.Linear(inshape, outshape)
        self.h1_dnn = nn.Linear(inshape, hidden_size)
        self.h2_dnn = nn.Linear(hidden_size, hidden_size)
        self.h3_dnn = nn.Linear(hidden_size, outshape)
        #self.amp_dnn = nn.Linear(inshape, )
        #self.norm = nn.LayerNorm([outshape])
        #if act is not None:
        nn.init.normal_(self.c_dnn.weight , mean = 0 , std = 0.01)
        nn.init.normal_(self.r_dnn.weight , mean = 0 , std = 0.01)
        nn.init.normal_(self.h1_dnn.weight , mean = 0 , std = 0.01)
        nn.init.normal_(self.h2_dnn.weight , mean = 0 , std = 0.01)
        nn.init.normal_(self.h3_dnn.weight , mean = 0 , std = 0.01)


        # else:
        #     nn.init.xavier_normal_(self.c_dnn.weight,gain=1.414)
        #     nn.init.xavier_normal_(self.r_dnn.weight, gain=1.414)
        #nn.init.normal_(self.t_dnn.weight , mean = 0 , std = 0.01)

        self.dp = nn.Dropout(p = 0.1)
        self.act = act
        self.norm = nn.BatchNorm1d([inshape])
    # def forward(self, kws):
    #     l, t = kws
    #     f_r, f_i = l * torch.cos(t), l * torch.sin(t)
    #     f_r, f_i = self.r_dnn(f_r), self.r_dnn(f_i)
        
    #     l = torch.log(l + 1e-6)
        
    #     l, t = self.c_dnn(l), self.c_dnn(t)
    #     l = torch.exp(l)

    #     r, i = l * torch.cos(t) + f_r, l * torch.sin(t) + f_i
    #     #r, i = torch.sigmoid(r), torch.sigmoid(i)
    #     #r, i = self.norm(r), self.norm(i)
    #    # r, i = self.dp(r), self.dp(i)
    #    # r, i = torch.relu(r), torch.relu(i)


    #     return (r**2 + i**2 + 1e-8) ** 0.5, t + torch.atan2(i, r)

    def forward(self, kws):
        f_r, f_i = kws


        #ffr, ffi = self.h1_dnn(f_r), self.h1_dnn(f_i)
        l = f_r ** 2 + f_i ** 2
        t = torch.atan2(f_i, f_r + 1e-3)

        f_r, f_i = self.dp(f_r), self.dp(f_i)
        f_r, f_i = self.r_dnn(f_r), self.r_dnn(f_i)
        if self.act:
             f_r, f_i = self.act(f_r), self.act(f_i)


        

        #l = ffr ** 2 + ffi ** 2
        #t = torch.atan2(ffi, ffr)  
        l = 0.5 * torch.log(l)
       # l = self.norm(l)
        l, t = self.dp(l), self.dp(t)
        #t = t / (2 * torch.pi)
        l, t = self.c_dnn(l), self.c_dnn(t)
        #l, t = self.h2_dnn(l), self.h2_dnn(t)
        #t = t * (2 * torch.pi)
        # if self.act:
        #     l, t = self.act(l), self.act(t)
        l = torch.exp(l)
        #ffr, ffi = l * torch.cos(t), l * torch.sin(t)

        #ffr, ffi = self.h3_dnn(ffr), self.h3_dnn(ffi)

        #r, i = ffr + f_r, ffi + f_i
        r, i = f_r + l * torch.cos(t), f_i + l * torch.sin(t)
        #r, i = torch.sigmoid(r), torch.sigmoid(i)
        #r, i = self.norm(r), self.norm(i)
       # r, i = self.dp(r), self.dp(i)
       # r, i = torch.relu(r), torch.relu(i)
        return r, i

    # def forward(self, kws):
    #     f_r, tha = kws

    #     l = f_r ** 2 + f_i ** 2
    #     t = torch.atan2(f_i, f_r)

    #     f_r, f_i = self.dp(f_r), self.dp(f_i)
    #     f_r, f_i = self.r_dnn(f_r), self.r_dnn(f_i)
    #     if self.act:
    #          f_r, f_i = self.act(f_r), self.act(f_i)

    #     l = 0.5 * torch.log(l)
    #    # l = self.norm(l)
        
    #     l, t = self.dp(l), self.dp(t)
    #     t = t % (2 * torch.pi)
    #     t = t / (2 * torch.pi)
    #     l, t = self.c_dnn(l), self.c_dnn(t)
    #     t = t * (2 * torch.pi)
    #     # if self.act:
    #     #     l, t = self.act(l), self.act(t)
    #     l = torch.exp(l)
    #     r, i = l * torch.cos(t) + f_r, l * torch.sin(t) + f_i
    #     #r, i = torch.sigmoid(r), torch.sigmoid(i)
    #     #r, i = self.norm(r), self.norm(i)
    #    # r, i = self.dp(r), self.dp(i)
    #    # r, i = torch.relu(r), torch.relu(i)
    #     return r, i

# 一份进block, 一份进deep，然后合并
class Bridge_BlockV1(nn.Module):
    def __init__(self, inshape, outshape, act = None) -> None:
        super().__init__()

        self.inshape, self.outshape = inshape, outshape

        self.c_dnn = nn.Linear(inshape // 16, outshape // 16)
        self.r_dnn = nn.Linear(inshape, outshape)

        #if act is not None:
        
        nn.init.normal_(self.c_dnn.weight , mean = 0.1 , std = 0.01)

        self.bias_lam = nn.Parameter(torch.randn(1, 16, outshape // 16) * 0.01)
        self.bias_tha = nn.Parameter(torch.randn(1, 16, outshape // 16) * 0.01)

        self.weight_lam = nn.Parameter(torch.randn(1, inshape // 16, 16) * 0.01)
        self.weight_tha = nn.Parameter(torch.randn(1, inshape // 16, 16) * 0.01)

        # by_tensor = (torch.randint(100,[outshape // 16, inshape // 16]) >= 99).float()
        with torch.no_grad():
            self.c_dnn.weight = nn.Parameter(torch.eye(self.inshape // 16))
        #nn.init.xavier_normal_(self.c_dnn.weight , gain=1.414)
        nn.init.normal_(self.r_dnn.weight , mean = 0 , std = 0.01)
        #nn.init.xavier_normal_(self.r_dnn.weight , gain=1.414)

        self.dp = nn.Dropout(p = 0.0)
        self.act = act
        self.norm = nn.BatchNorm1d([inshape])

        self.reversed = [self.c_dnn]

    def forward(self, kws):
        f_r, f_i = kws


        #ffr, ffi = self.h1_dnn(f_r), self.h1_dnn(f_i)
        l = f_r ** 2 + f_i ** 2 + 1e-6
        t = torch.atan2(f_i + 1e-6, f_r + 1e-6)

        f_r, f_i = f_r.view(f_r.shape[0], -1), f_i.view(f_i.shape[0], -1)
        f_r, f_i = self.dp(f_r), self.dp(f_i)
        f_r, f_i = self.r_dnn(f_r), self.r_dnn(f_i)
        if self.act:
             f_r, f_i = self.act(f_r), self.act(f_i)
        f_r, f_i = f_r.view(f_r.shape[0], -1, 16), f_i.view(f_r.shape[0], -1, 16)

        l = 0.5 * torch.log(l+ 1e-6)
        l, t = l + self.weight_lam, t + self.weight_tha
        l, t = self.dp(l), self.dp(t)
        l, t = torch.transpose(l, -2, -1), torch.transpose(t, -2, -1)
        l, t = self.c_dnn(l) + self.bias_lam, self.c_dnn(t) + self.bias_tha
        l = torch.exp(l)
        l, t = torch.transpose(l, -2, -1), torch.transpose(t, -2, -1)

        r, i = f_r + l * torch.cos(t), f_i + l * torch.sin(t)
        return r, i


class CMLP(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        self.mlplist = config.mlp
        #self.amp_dnn = DNN(config , [512, 128], autoMatch= False)
        self.dnn = DNN(config , [512, 758])
        self.tha_dnn = DNN(config, [758, 512, 1], autoMatch= False)
        self.backbone = ['dnn']
        self.b_log = torch.nn.LayerNorm([(len(config.feature_stastic) - 1), config.embedding_size])
      

    def FeatureInteraction(self , feature , sparse_input, *kargs):
        lam, tha = torch.sigmoid(feature), kargs[0]
        #lam = f_r / torch.cos(tha)
        #tha = torch.arccos((f_r) / (lam + 1e-6)) * torch.sign(f_i + 1e-6)

        #tha = tha.view(tha.shape[0],)
        lam = torch.log(lam)
        # lam = self.b_log(lam)

        lam, tha = self.dnn(lam), self.dnn(tha)

        
        lam = torch.exp(lam)

        f_r, f_i = lam * torch.cos(tha), lam * torch.sin(tha)

        f_r, f_i = self.tha_dnn(f_r), self.tha_dnn(f_i)

        self.logits = f_r + f_i
        self.ouput = torch.sigmoid(self.logits)
        return self.ouput

class DMLP(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        self.mlplist = [16 * 23, 1024, 1024, 1]
        mlps = []

        for inshape, outshape in zip(self.mlplist[:-1], self.mlplist[1:]):
            mlps.append(Bridge_Block(inshape, outshape, nn.ReLU() if outshape != 1 else None))

        self.bkb = nn.Sequential(*mlps)

    def FeatureInteraction(self , feature , sparse_input, *kargs):
        #lam, tha = torch.sigmoid(feature), kargs[0]
        lam, tha = feature, kargs[0]
        lam, tha = lam.view(lam.shape[0], -1), tha.view(tha.shape[0], -1)
        lam, tha = self.bkb((lam, tha))

        self.logits = lam  + tha #lam * torch.cos(tha) + lam * torch.sin(tha)
        self.ouput = torch.sigmoid(self.logits)
        return self.ouput

class EMLP(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        field_num = len(config.feature_stastic) - 1
        self.mlplist = [16 * field_num, 16 * field_num, 16 * field_num, 16 * field_num, 16 * field_num]
        mlps = []

        idx = -1
        for inshape, outshape in zip(self.mlplist[:-1], self.mlplist[1:]):
            idx += 1
            mlps.append(Bridge_BlockV1(inshape, outshape, nn.ReLU() if idx != len(self.mlplist) - 2 else None))

        self.tha_dnn = DNN(config, [0, 512, 512, 1])
        self.bkb = nn.Sequential(*mlps)

        self.regular = nn.Linear(self.mlplist[-1], 1)
        #nn.init.normal_(self.regular.weight, mean = 0, std =)
        # self.reserved = []
        # for modu in mlps:
        #     self.reserved.extend()

    def FeatureInteraction(self , feature , sparse_input, *kargs):
        #lam, tha = torch.sigmoid(feature), kargs[0]
        r, i = feature, kargs[0]
        r, i = self.bkb((r, i))
        r, i = r.view(r.shape[0], -1), i.view(i.shape[0], -1)
        r, i = self.regular(r), self.regular(i)
        self.logits =r + i #lam * torch.cos(tha) + lam * torch.sin(tha)
        self.ouput = torch.sigmoid(self.logits)
        return self.ouput

class MLP(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        self.mlplist = config.mlp
        self.dnn = DNN(config , self.mlplist, dp_rate= 0.2, drop_last= True)
        self.backbone = ['dnn']
        field_num = self.field_num = len(config.feature_stastic) - 1
        self.C = nn.Parameter(torch.ones(1, field_num, 1))
        self.p = DNN(config, [16, 16], autoMatch= False, drop_last= False)

    def FeatureInteraction(self , feature , sparse_input, *kargs):
        self.logits = self.dnn(feature)
        self.output = torch.sigmoid(self.logits)
        return self.output

class ComplexMLPs(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        self.mlplist = config.mlp
        self.dnn = DNN(config , self.mlplist, dp_rate= 0.2, drop_last= True)
        self.backbone = ['dnn']
        field_num = self.field_num = len(config.feature_stastic) - 1
        self.C = nn.Parameter(torch.ones(1, field_num, 1))
        self.p = DNN(config, [16, 16], autoMatch= False, drop_last= False)

    def FeatureInteraction(self, feature , sparse_input, *kargs):
        r, i = self.C * torch.cos(feature), self.C * torch.sin(feature)
        self.logits = self.dnn(r) + self.dnn(i)
        self.output = torch.sigmoid(self.logits)
        return self.output

class InterMLP(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        self.mlplist = config.mlp
        self.dnn = DNN(config , self.mlplist, dp_rate= 0.3, drop_last= True)
        self.backbone = ['dnn']
        field_num = self.field_num = len(config.feature_stastic) - 1
        self.C = nn.Parameter(torch.ones(1, field_num, 1))
        self.p = DNN(config, [16, 8, 1], autoMatch= False, drop_last= True, dp_rate=0.)
        self.dp = nn.Dropout(p = 0.3)

    def FeatureInteraction(self , feature , sparse_input, *kargs):
        r, i = self.C * torch.sin(self.p(feature) * feature + torch.pi / 2), self.C * torch.sin(self.p(feature) * feature)
        r, i = self.dp(r), self.dp(i)
        self.logits = self.dnn(r) + self.dnn(i)
        self.output = torch.sigmoid(self.logits)
        return self.output

from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
import torch
from torch import nn
layer = 0
class EG(nn.Module):
    def __init__(self, config, inshape, outshape, act = None, type = 'explicit') -> None:
        super().__init__()
        global layer
        layer += 1
        self.inshape, self.outshape = inshape, outshape
        self.group_size = config.embedding_size
#        self.c_dnn = nn.Linear(inshape // self.group_size, outshape // self.group_size)
        xx = torch.eye(inshape // self.group_size, outshape // self.group_size)
        # idx = torch.arange(len(xx))
        # xx[idx, (idx + len(xx)) % len(xx)] -= .5
        self.c_dnn = nn.Parameter(xx)
        #self.c_dnn.requires_grad_(False)
        #nn.init.xavier_normal_(self.c_dnn, gain = 1.414)
        self.r_dnn = nn.Linear(inshape, outshape)
        self.bias_lam = nn.Parameter(torch.randn(1, self.group_size, outshape // self.group_size) * 0.01)
        self.bias_tha = nn.Parameter(torch.randn(1, self.group_size, outshape // self.group_size) * 0.01)

        #self.weight_lam = nn.Parameter(torch.randn(1, inshape // self.group_size, 1) * 0.01)
        #self.weight_tha = nn.Parameter(torch.randn(1, inshape // self.group_size, 1) * 0.01)

        # by_tensor = (torch.randint(100,[outshape // 16, inshape // 16]) >= 99).float()
        # with torch.no_grad():
        #     self.c_dnn = nn.Parameter(torch.eye(self.inshape // self.group_size))
        #nn.init.xavier_normal_(self.c_dnn , gain=1.414)
        nn.init.normal_(self.r_dnn.weight , mean = 0 , std = 0.01)
        #nn.init.xavier_normal_(self.r_dnn.weight , gain=1.414)

        #self.field_weight = nn.Parameter(torch.randn(1,  outshape // self.group_size, self.group_size) * 0.01)

        self.dp = nn.Dropout(p = config.dp_rate)
        self.act = act

        # self.norm = nn.BatchNorm1d([outshape // self.group_size])
        # self.norm1 = nn.BatchNorm1d([outshape // self.group_size])
        # self.norm = nn.LazyBatchNorm1d()
        # self.norm1 = nn.LazyBatchNorm1d()
        self.norm = nn.LayerNorm([inshape // self.group_size, self.group_size])
        self.norm1 = nn.LayerNorm([inshape // self.group_size, self.group_size])
        self.reversed = [self.c_dnn]
        
        self.order = nn.Parameter(torch.ones(7,1))
        
        self.type = type


    def forward(self, kws):
        f_r, f_i = kws
        #f_r, f_i = self.norm(f_r), self.norm1(f_i)
        rfr, rfi =  f_r, f_i
        #rfr, rfi = rfr + self.weight_lam, rfi + self.weight_tha
        l = rfr ** 2 + rfi ** 2 + 1e-8

        t = torch.atan2(rfi, rfr)
        l, t = l.reshape(l.shape[0], -1, self.group_size), t.reshape(t.shape[0], -1, self.group_size)

        f_r, f_i = f_r.reshape(f_r.shape[0], -1), f_i.reshape(f_i.shape[0], -1)
        f_r, f_i = self.dp(f_r), self.dp(f_i)
        f_r, f_i = self.r_dnn(f_r), self.r_dnn(f_i)
        if self.act:
             f_r, f_i = self.act(f_r), self.act(f_i)
        f_r, f_i = f_r.reshape(f_r.shape[0], -1, self.group_size ), f_i.reshape(f_r.shape[0], -1, self.group_size )

        l = 0.5 * torch.log(l)
        l, t = torch.transpose(l, -2, -1), torch.transpose(t, -2, -1)
        l, t = self.dp(l), self.dp(t)
        l, t =  l @ (self.c_dnn) + self.bias_lam,  t @ (self.c_dnn) + self.bias_tha
        l = torch.exp(l)
        l, t = torch.transpose(l, -2, -1), torch.transpose(t, -2, -1)

        r, i =   l * torch.cos(t) + f_r,\
                l * torch.sin(t) + f_i
        r, i = r.reshape(r.shape[0], -1, self.group_size ), i.reshape(i.shape[0], -1, self.group_size )
        return r, i

import numpy as np
class EulerNet(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        field_num = self.field_num = len(config.feature_stastic) - 1
        self.mlplist = [config.embedding_size * field_num] * (config.num_layers + 1)#[16 * field_num,  16 * field_num, 16 * field_num, 16 * field_num, 16 * field_num]
        #self.mlplist[1] = 128 * field_num
        #self.mlplist = (np.array([field_num] + config.mlplist) * 16).tolist()
        mlps = []

        idx = -1
        for inshape, outshape in zip(self.mlplist[:-1], self.mlplist[1:]):
            idx += 1
            mlps.append(EG(config, inshape, outshape, nn.ReLU(), 'implicit' if idx < 1 else 'implicit'))

        #self.tha_dnn = DNN(config, [0, 512, 512, 1])
        self.bkb = nn.Sequential(*mlps)
        self.C = nn.Parameter(torch.ones(1, field_num, 1))
        self.regular = nn.Linear(self.mlplist[-1], 1)
        #nn.init.normal_(self.regular.weight, mean = 0, std =)
        # self.reserved = []
        # for modu in mlps:
        #     self.reserved.extend()

    def FeatureInteraction(self , feature , sparse_input, *kargs):
        #lam, tha = torch.sigmoid(feature), kargs[0]
        #r, i = torch.abs(feature), torch.zeros_like(feature)
        #r, i = feature, torch.zeros_like(feature)#, kargs[0]
        #r, i = feature, kargs[0]
        # torch.cos(x)
        r, i = self.C * torch.cos(feature), self.C * torch.sin(feature)
        r, i = self.bkb((r, i))
        r, i = r.reshape(r.shape[0], -1), i.reshape(i.shape[0], -1)
        r, i = self.regular(r), self.regular(i)
        self.logits =r + i #lam * torch.cos(tha) + lam * torch.sin(tha)
        self.ouput = torch.sigmoid(self.logits)
        return self.ouput



class EulerNetE(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        field_num = self.field_num = len(config.feature_stastic) - 1
        self.mlplist = [16 * field_num, 16 * field_num, 16 * field_num, 16 * field_num]#[16 * field_num,  16 * field_num, 16 * field_num, 16 * field_num, 16 * field_num]
        mlps = []
        idx = -1
        for inshape, outshape in zip(self.mlplist[:-1], self.mlplist[1:]):
            idx += 1
            mlps.append(EG_Implicit(inshape, outshape, nn.ReLU(), 'implicit' if idx < 1 else 'implicit'))
        self.imp_bkb = nn.Sequential(*mlps)

        self.mlplist = [16 * field_num,  16 * field_num]#[16 * field_num,  16 * field_num, 16 * field_num, 16 * field_num, 16 * field_num]
        mlps = []
        idx = -1
        for inshape, outshape in zip(self.mlplist[:-1], self.mlplist[1:]):
            idx += 1
            mlps.append(EG_Explicit(inshape, outshape, nn.ReLU(), 'implicit' if idx < 1 else 'implicit'))
        self.exp_bkb = nn.Sequential(*mlps)

        self.i_regular = nn.Linear(self.mlplist[-1], 1)
        self.e_regular = nn.Linear(self.mlplist[-1], 1)
        self.regular = nn.Linear(self.mlplist[-1], 1)
        #nn.init.normal_(self.regular.weight, mean = 0, std =)
        # self.reserved = []
        # for modu in mlps:
        #     self.reserved.extend()

    def FeatureInteraction(self , feature , sparse_input, *kargs):
        #lam, tha = torch.sigmoid(feature), kargs[0]
        r, i = torch.cos(feature), torch.sin(feature)#, kargs[0]
        ir, ii = self.imp_bkb((r, i))
        er, ei = self.exp_bkb((r,i))
        # ir, ii = ir.reshape(r.shape[0], -1), ii.reshape(r.shape[0], -1)
        # er, ei = er.reshape(r.shape[0], -1), ei.reshape(r.shape[0], -1)
        r, i = (ir + er).view(r.shape[0], -1), (ii + ei).view(i.shape[0], -1)
        r, i = self.regular(r), self.regular(i)
        self.logits = r + i#self.i_regular(ir) + self.i_regular(ii) + self.e_regular(er) + self.e_regular(ei) #lam * torch.cos(tha) + lam * torch.sin(tha)
        self.ouput = torch.sigmoid(self.logits)
        return self.ouput


class EulerNetBT(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        field_num = self.field_num = len(config.feature_stastic) - 1
        self.mlplist = [16 * field_num, 16 * field_num]#[16 * field_num,  16 * field_num, 16 * field_num, 16 * field_num, 16 * field_num]
        mlps = []

        idx = -1
        for inshape, outshape in zip(self.mlplist[:-1], self.mlplist[1:]):
            idx += 1
            mlps.append(EG(inshape, outshape, nn.ReLU()))

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

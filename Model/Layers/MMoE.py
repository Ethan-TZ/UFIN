import torch
from torch import nn
from torch.nn import functional as F

Emd_Size = 16

class PWLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()
    
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)

    def forward(self, x):
        xx = self.lin(self.dropout(x) - self.bias)
        return xx

class MoEAdaptorLayer(nn.Module):
    def __init__(self, n_exps, layers, dropout=0.0, noise=False):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)

class EG(nn.Module):
    def __init__(self, inshape, outshape, drop_out = 0.3,  act = None, type = 'explicit') -> None:
        super().__init__()
        self.inshape, self.outshape = inshape, outshape
        self.group_size = Emd_Size
        xx = torch.eye(inshape // self.group_size, outshape // self.group_size)
        self.c_dnn = nn.Parameter(xx)
        self.r_dnn = nn.Linear(inshape, outshape)
        self.re_connect = nn.Parameter(torch.eye(inshape // 16, inshape //16) )
        self.bias_lam = nn.Parameter(torch.randn(1, self.group_size, outshape // self.group_size) * 0.01)
        self.bias_tha = nn.Parameter(torch.randn(1, self.group_size, outshape // self.group_size) * 0.01)
        nn.init.normal_(self.r_dnn.weight , mean = 0 , std = 0.01)
        self.dp = nn.Dropout(p = drop_out)
        self.act = act
        self.reversed = [self.c_dnn]
        self.order = nn.Parameter(torch.ones(7,1))
        self.type = type
        self.norm = nn.LayerNorm([self.group_size])
        self.norm1 = nn.LayerNorm([self.group_size])

    def forward(self, kws):
        f_r, f_i = kws
        rfr, rfi =  f_r, f_i
        l = rfr ** 2 + rfi ** 2 + 1e-8

        t = torch.atan2(rfi, rfr + 1e-8)
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


class EulerInteraction(nn.Module):
    def __init__(self , input_dim, field_num, drop_out = 0.3) -> None:
        super().__init__()
        self.mlplist = [input_dim * field_num] * 2
        mlps = []

        idx = -1
        for inshape, outshape in zip(self.mlplist[:-1], self.mlplist[1:]):
            idx += 1
            mlps.append(EG(inshape, outshape,  drop_out= drop_out, act= nn.ReLU(), type = 'implicit' if idx < 1 else 'implicit'))

        self.bkb = nn.Sequential(*mlps)
        self.C = nn.Parameter(torch.ones(1, field_num, 1))
        self.regular = nn.Linear(self.mlplist[-1], 1)

    def forward(self , feature):
        r, i = self.C * torch.cos(feature), self.C * torch.sin(feature)
        r, i = self.bkb((r, i))
        r, i = r.reshape(r.shape[0], -1), i.reshape(i.shape[0], -1)
        self.sta = r + i
        r, i = self.regular(r), self.regular(i)
        return r + i

class EulerMoE(nn.Module):
    def __init__(self, n_exps, layers, dropout=0.0, noise=False):
        super(EulerMoE, self).__init__()
        self.n_exps = n_exps
        self.noisy_gating = noise
        self.W = nn.Linear(layers[0], layers[0], False)
        self.experts = nn.ModuleList([EulerInteraction(Emd_Size, layers[0] // Emd_Size, drop_out= dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.norm = nn.LayerNorm([Emd_Size])
        with torch.no_grad():
            self.W.weight = nn.Parameter(torch.eye(layers[0]))
        self.KG = torch.eye(self.n_exps).cuda()

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        _, p = torch.topk(logits, 7, dim = -1) # bk
        p = self.KG[p] #bke
        p = (torch.sum(p, dim = 1)).bool().float() # be
        gates = F.softmax(logits, dim=-1)
        return gates * p

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        x = self.W(x)
        x = x.reshape(x.shape[0], -1, Emd_Size)
        x = self.norm(x)
        self.stb = x
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        
        sta = [self.experts[i].sta.unsqueeze(-2) for i in range(self.n_exps)]
        sta = torch.cat(sta, dim = -2)
        multiple_sta = gates.unsqueeze(-1) * sta
        self.sta = multiple_sta.sum(dim = -2)

        return multiple_outputs.sum(dim=-2)

import torch
from torch import nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
from Model.Layers.mDNN  import mDNN
 # res or  stack mlp

class PN(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, num_scenes = 7):
        super(PN, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.scene_val = nn.Parameter(torch.ones(num_scenes, num_features))
        self.scene_mean = nn.Parameter(torch.zeros(num_scenes, num_features))

    def forward(self, input):                            
        input, scene = input
        p_val, p_mean = self.scene_val[scene.long()][:,:,None], self.scene_mean[scene.long()][:,:,None]
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:   
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1           
                if self.momentum is None:                
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:                                    
                    exponential_average_factor = self.momentum

        if self.training:
            mean = input.mean([0, 2])           
            var = input.var([0, 2], unbiased=False)    
            n = input.numel() / input.size(1)            
            with torch.no_grad():                         
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
            mean = self.running_mean
            var = self.running_var
        input = (input - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            input = input * (self.weight[None, :, None] * p_val) + self.bias[None, :, None] + p_mean

        return input


class STARLinear(nn.Module):
    def __init__(self, inshape, outshape, num_scenes, dp_rate = 0.1, drop = False) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_scenes, inshape, outshape))
        self.b = nn.Parameter(torch.zeros(num_scenes, outshape))

        self.global_W = nn.Parameter(torch.randn(1, inshape, outshape))
        self.global_b = nn.Parameter(torch.zeros(1, outshape))
        self.drop = drop
        self.drop_out = nn.Dropout(p = dp_rate)
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.global_W)
    
    def forward(self, input):
        x, scene = input
        w, b = self.W[scene.long() - 1], self.b[scene.long() - 1] #[b,]
        x =  (x[:,None,:]@(w * self.global_W)).squeeze(1) + b + self.global_b
        if not self.drop:
            x = torch.relu(x)
        x = self.drop_out(x)
        return x, scene


class STARDNN(nn.Module):
    def __init__(self , config:Config , Shape , drop_last = True , act = None , autoMatch = True, dp_rate = 0.1):
        super().__init__()
        layers = []
        self.autoMatch = autoMatch
        if self.autoMatch:
            Shape[0] = (len(config.feature_stastic) -1 ) * config.embedding_size
        for i in range(0 , len(Shape) - 2):
            hidden = STARLinear(Shape[i] , Shape[i + 1] , 7, dp_rate, False)
            layers.append(hidden)
        layers.append(STARLinear(Shape[-2] , Shape[-1] , 7, dp_rate, drop= True))
        self.net = nn.Sequential(*layers)
    
    def forward(self , input):
        x, scene = input
        if self.autoMatch:
            x = x.reshape(x.shape[0] , -1)
        return self.net((x, scene))


class STAR(BasicModel):
    def __init__(self, config : Config):
        super().__init__(config)
        self.top_mlp = STARDNN(config, [128, 256, 256, 256, 1], dp_rate = 0.1)
        self.pn = PN(num_features= len(config.feature_stastic) -1)
        self.scene_embedding = nn.Embedding(8, 16)
        nn.init.xavier_uniform(self.scene_embedding.weight)
        self.aux_mlp = DNN(config, [128, 256, 256, 1],autoMatch= False)

    def FeatureInteraction(self , feature , sparse_input, *kwards):
        x, scene = feature, sparse_input['scene']
        # x = self.pn([x, scene])
        sc_emd = self.scene_embedding(scene.long().cuda())
        aux_input = torch.cat([x, sc_emd[:,None,:]], dim = 1).view(x.shape[0], -1)
        x, scene = self.top_mlp([x, scene])
        self.logits = x + self.aux_mlp(aux_input)
        self.output = torch.sigmoid(self.logits)
        return self.output
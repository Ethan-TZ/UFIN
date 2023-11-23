from operator import index
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
import torch
from torch import nn
from Model.PLM import PLM
from torch.nn import functional as F
from Model.Layers.MMoE import MoEAdaptorLayer, CRSMoE, EulerMoE, EulerInteraction, CINMoE

class UFIN(BasicModel):
    def __init__(self , config: Config, phase = 'training') -> None:
        super().__init__(config)
        self.config = config
        self.plm_shape = config.plmshape
        self.num_experts = config.num_experts
        hidden_size = (len(config.feature_stastic) -1 ) * config.embedding_size
        self.llm_moe = MoEAdaptorLayer(self.num_experts, [self.plm_shape, self.plm_shape])
        self.moe = EulerMoE(self.num_experts, [self.plm_shape])
        self.fine_tune_adaptiver = nn.Linear(hidden_size, 1, bias = False)
        self.phase = phase
        self.MSELoss = nn.MSELoss()
        self.ft_euler = EulerInteraction(config.embedding_size, hidden_size // config.embedding_size, 0.0)
        self.U = nn.Linear(config.embedding_size, self.plm_shape, bias= False)
        nn.init.xavier_normal_(self.U.weight)
        self.norm = nn.LayerNorm([self.plm_shape])
        self.is_add_fa = config.is_add_fa

    def forward(self , sparse_input, dense_input = None):
        predict = self.FeatureInteraction(None , sparse_input, self.i_embedding_layer(sparse_input, "no_id"), None)
        return predict

    def FeatureInteraction(self , feature , sparse_input, *kargs):
        if not ('encoding' in sparse_input):
            if not hasattr(self,'plm'):
                self.plm = PLM(self.config)
            plm_out = self.plm(sparse_input)
        else:
            plm_out = sparse_input['encoding'].cuda()

        plm_out = self.norm(plm_out)
        plm_out = self.llm_moe(plm_out)

        faker = plm_out + self.U(kargs[0].view(kargs[0].shape[0], -1))
        faker = self.moe(faker)

        if self.is_add_fa:
            self.logits = faker * 0.4 + self.ft_euler(feature)
        else:
            self.logits = faker

        self.output = torch.sigmoid(self.logits)
        if 'teacher_logits' in sparse_input:
            self.aux_loss += self.MSELoss(sparse_input['teacher_logits'].cuda(), faker)
        return self.output
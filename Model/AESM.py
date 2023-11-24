import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation
from Model.BasicModel import BasicModel

class MMoE_Layer(nn.Module):
    def __init__(self, num_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(MMoE_Layer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_experts)])

        self.gate = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                             hidden_units=gate_hidden_units,
                                             output_dim=num_experts,
                                             hidden_activations=hidden_activations,
                                             output_activation=None,
                                             dropout_rates=net_dropout,
                                             batch_norm=batch_norm) for _ in range(self.num_tasks)])
        self.gate_activation = get_activation('softmax')

        self.S = nn.Parameter(torch.randn(7, input_dim + 16, self.num_experts) * 0.01)
        self.scene_emb = nn.Embedding(10, 16)

        self.KG = torch.eye(self.num_experts).cuda()
        nn.init.xavier_normal_(self.S)

    def forward(self, x, scene):
        self.aux_loss = 0
        experts_output = torch.stack([self.experts[i](x) for i in range(self.num_experts)],
                                     dim=1)  # (?, num_experts, dim)
        # K, S
        scene_emb = self.scene_emb(scene.long().cuda())
        x = torch.cat([x, scene_emb], dim = -1)
        G = torch.einsum('bd,sde->bse', x, self.S).transpose(-2, -1) # bes
        G = torch.softmax(G, dim = -1) # bes
        p = torch.log(G) # bes
        q = torch.sum(torch.log(G * 7) / 7, dim = -1) # be
        self.aux_loss += 1e-3 * (-torch.sum(p) - torch.sum(q))

        ses = self.KG[scene.long().cuda()].unsqueeze(1) # b1s
        
        _, p = torch.topk(torch.sum(ses * p, dim = -1), 6, dim = -1) # bk
        _, q = torch.topk(q, 6, dim = -1) # bk

        p = self.KG[p]
        q = self.KG[q]
        select = torch.cat([p,q], dim  = 1)
        select = torch.sum(select, dim = 1).bool().float()

        mmoe_output = []
        for i in range(self.num_tasks):
            gate_output = torch.sum(G * ses, dim = -1)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)  # (?, num_experts)
            gate_output = gate_output * select
            mmoe_output.append(torch.sum(torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1))
        return mmoe_output

    def get_aux(self):
        return self.aux_loss 

class AESM(BasicModel):
    def __init__(self, config):
        super(AESM, self).__init__(config)
                #  task=["binary_classification"],
                #  num_tasks=1,
                #  model_id="MMoE",
                #  gpu=-1,
                #  learning_rate=1e-3,
                #  embedding_dim=10,
                #  num_experts=4,
                #  expert_hidden_units=[512, 256, 128],
                #  gate_hidden_units=[128, 64],
                #  tower_hidden_units=[128, 64],
                #  hidden_activations="ReLU",
                #  net_dropout=0,
                #  batch_norm=False,
                #  embedding_regularizer=None,
                #  net_regularizer=None,
        num_fields = len(config.feature_stastic) - 1
        embedding_dim = config.embedding_size
        self.num_tasks = 1
        self.mmoe_layer = MMoE_Layer(num_experts=config.num_experts,
                                     num_tasks=config.num_tasks,
                                     input_dim=embedding_dim * num_fields,
                                     expert_hidden_units=config.expert_hidden_units,
                                     gate_hidden_units=config.gate_hidden_units,
                                     hidden_activations="ReLU",
                                     net_dropout=0.,
                                     batch_norm=True)
        self.tower = nn.ModuleList([MLP_Block(input_dim=config.expert_hidden_units[-1],
                                              output_dim=1,
                                              hidden_units=config.tower_hidden_units,
                                              hidden_activations="ReLU",
                                              output_activation=None,
                                              dropout_rates=0.1,
                                              batch_norm=True)
                                    for _ in range(self.num_tasks)])
        


    def FeatureInteraction(self, feature, sparse_input, *kargs):
        x, scene = feature, sparse_input['scene']
        expert_output = self.mmoe_layer(x.flatten(start_dim=1), scene)
        tower_output = [self.tower[i](expert_output[i]) for i in range(self.num_tasks)]
        y_pred = [torch.sigmoid(tower_output[i]) for i in range(self.num_tasks)]

        return y_pred[0]

    def RegularLoss(self, weight):
        return self.mmoe_layer.get_aux()
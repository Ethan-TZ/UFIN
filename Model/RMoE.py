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

    def forward(self, x):
        experts_output = torch.stack([self.experts[i](x) for i in range(self.num_experts)],
                                     dim=1)  # (?, num_experts, dim)
        mmoe_output = []
        for i in range(self.num_tasks):
            gate_output = self.gate[i](x)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)  # (?, num_experts)
            mmoe_output.append(torch.sum(torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1))
        return mmoe_output


class RMoE(BasicModel):
    def __init__(self, config):
        super(RMoE, self).__init__(config)
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


    def FeatureInteraction(self, feature, *kargs):
        feature_emb = feature
        expert_output = self.mmoe_layer(feature_emb.flatten(start_dim=1))
        tower_output = [self.tower[i](expert_output[i]) for i in range(self.num_tasks)]
        y_pred = [torch.sigmoid(tower_output[i]) for i in range(self.num_tasks)]

        return y_pred[0]
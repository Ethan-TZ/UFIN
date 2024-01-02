# =========================================================================
# Copyright (C) 2022. FuxiCTR Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from Model import BasicModel
import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_activation


class PPNet(BasicModel):
    def __init__(self, config):
        super(PPNet, self).__init__(config)
        emd = config.embedding_size * (len(config.feature_stastic) - 1)
        gate_input_dim = emd
        self.ppn = PPNet_MLP(input_dim=emd,
                             output_dim=1,
                             gate_input_dim=gate_input_dim * 2,
                             gate_hidden_dim=config.gate_hidden_dim,
                             hidden_units=config.hidden_units,
                             hidden_activations="ReLU",
                             dropout_rates=0.,
                             batch_norm=False)
            
    def FeatureInteraction(self, feature, sparse_input, *kargs):
        feature_emb = feature.view(feature.shape[0], -1)
        gate_emb = kargs[1].view(feature.shape[0], -1)
        y_pred = self.ppn(feature_emb, gate_emb)
        y_pred = torch.sigmoid(y_pred)
        return y_pred


class PPNet_MLP(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim=1,
                 gate_input_dim=64,
                 gate_hidden_dim=None,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 dropout_rates=0.0,
                 batch_norm=True,
                 use_bias=True):
        super(PPNet_MLP, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        self.gate_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            layers = [nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx] is not None:
                layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                layers.append(nn.Dropout(p=dropout_rates[idx]))
            self.mlp_layers.append(nn.Sequential(*layers))
            self.gate_layers.append(GateNU(gate_input_dim, gate_hidden_dim, 
                                           output_dim=hidden_units[idx + 1]))
        self.mlp_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
    
    def forward(self, feature_emb, gate_emb):
        gate_input = torch.cat([feature_emb.detach(), gate_emb], dim=-1)
        h = feature_emb
        for i in range(len(self.gate_layers)):
            h = self.mlp_layers[i](h)
            g = self.gate_layers[i](gate_input)
            h = h * g
        out = self.mlp_layers[-1](h)
        return out


class GateNU(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 hidden_activation="ReLU",
                 dropout_rate=0.1):
        super(GateNU, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        layers = [nn.Linear(input_dim, hidden_dim)]
        layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.gate = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.gate(inputs) * 2
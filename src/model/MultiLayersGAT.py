import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from model.data import Arxiv


class MultiLayersGAT(nn.Module):
    def __init__(
            self, 
            in_feats, hidden_feats_list, num_heads_list, n_classes, 
            input_dropout, dropout_list):
        super(MultiLayersGAT, self).__init__()

        self.n_layers = len(hidden_feats_list) + 1
        self.convs = nn.ModuleList([
            dgl.nn.GATConv(
                in_feats if i == -1 else hidden_feats_list[i] * num_heads_list[i],
                n_classes if i == self.n_layers - 2 else hidden_feats_list[i + 1],
                1 if i == self.n_layers - 2 else num_heads_list[i + 1])
            for i in range(-1, self.n_layers - 1)])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(
                n_classes if i == self.n_layers - 1  else hidden_feats_list[i] * num_heads_list[i])
            for i in range(0, self.n_layers)])
        self.input_dropout = nn.Dropout(input_dropout)
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_list[i]) for i in range(0, self.n_layers)])
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, graph, x):
        x = self.input_dropout(x)

        for i in range(0, self.n_layers):
            x = self.convs[i](graph, x)
            x = self.flatten(x)
            x = self.norms[i](x)
            x = self.dropouts[i](x)
            x = F.relu(x)

        y = F.softmax(x, dim=-1)

        return y


if __name__ == "__main__":
    arxiv = Arxiv()
    print(arxiv.g.ndata['feat'].shape[-1])
    model = MultiLayersGAT(
            arxiv.g.ndata['feat'].shape[-1], [256, 256], [3, 3], 40,
            0.3, [0.1, 0.1, 0.1])
    predictions = model(arxiv.g, arxiv.g.ndata['feat'])
    print(predictions.shape)


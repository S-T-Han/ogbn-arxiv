import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from data import Arxiv


class MultiLayersGCN(nn.Module):
    def __init__(
            self, 
            in_feats, hidden_feats_list, n_classes, 
            dropout, use_linear):
        super(MultiLayersGCN, self).__init__()

        self.n_layers = len(hidden_feats_list) + 1
        self.convs = nn.ModuleList([
            dgl.nn.GraphConv(
                in_feats if i == -1 else hidden_feats_list[i],
                n_classes if i == self.n_layers - 2 else hidden_feats_list[i + 1])
            for i in range(-1, self.n_layers - 1)])
        self.linears = nn.ModuleList([
            nn.Linear(
                in_feats if i == -1 else hidden_feats_list[i],
                n_classes if i == self.n_layers - 2 else hidden_feats_list[i + 1],
                bias=False)
            for i in range(-1, self.n_layers - 1)]) if use_linear else None
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(
                n_classes if i == self.n_layers - 1  else hidden_feats_list[i])
            for i in range(0, self.n_layers)])
        self.input_dropout = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, x):
        x = self.input_dropout(x)

        if self.linears is not None:
            for i in range(0, self.n_layers):
                x = self.convs[i](graph, x) + self.linears[i](x)
                x = self.norms[i](x)
                x = F.relu(x)
        else:
            for i in range(0, self.n_layers):
                x = self.convs[i](graph, x)
                x = self.norms[i](x)
                x = F.relu(x)

        y = F.softmax(x, dim=-1)

        return y


if __name__ == "__main__":
    i = -1
    l = [1, 2, 3]
    l = [-2 if i == -1 else l[i] for i in range(-1, 3)]

    arxiv = Arxiv()
    print(arxiv.g.ndata['feat'].shape[-1])
    model = MultiLayersGCN(
            arxiv.g.ndata['feat'].shape[-1], [256, 256], 40,
            0.3, True)
    predictions = model(arxiv.g, arxiv.g.ndata['feat'])
    print(predictions.shape)

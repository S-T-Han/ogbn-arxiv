import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.dataloading import NodeDataLoader, NeighborSampler
from model.data import Arxiv


class TwoLayersGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, label_size):
        super(TwoLayersGCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dgl.nn.GraphConv(hidden_feats, out_feats)
        self.linear = nn.Linear(out_feats, label_size)

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = F.relu(x)
        x = self.conv2(blocks[1], x)
        x = F.relu(x)
        x = self.linear(x)
        y = F.softmax(x, dim=-1)

        return y


if __name__ == "__main__":
    arxiv = Arxiv()

    sampler = NeighborSampler([20, 20])
    dataloader = NodeDataLoader(
            arxiv.g, arxiv.idx['train'], sampler, 
            batch_size=256, 
            shuffle=True, 
            num_workers=4)

    input_nids, output_nids, blocks = next(iter(dataloader))
    output_labels = arxiv.labels[output_nids]

    model = TwoLayersGCN(arxiv.g.ndata['feat'].shape[-1], 128, 64, 40)
    pridections = model(blocks, blocks[0].srcdata['feat'])
    print(pridections.shape)
    print(torch.sum(pridections[0]))

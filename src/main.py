import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import dgl
from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler, NodeDataLoader
from ogb.nodeproppred import Evaluator

from model.data import Arxiv
from model.TwoLayersGCN import TwoLayersGCN
from model.TwoLayersSAGE import TwoLayersSAGE
from model.MultiLayersGCN import MultiLayersGCN


def train_blocks(model, loss_fn, optimizer, dataloader, labels, device):
    assert isinstance(model, torch.nn.Module)
    model.train()
    for i, (input_nids, output_nids, blocks) in enumerate(dataloader):
        blocks = [block.to(device) for block in blocks]
        output_labels = labels[output_nids].flatten().to(device)
        input_feats = blocks[0].srcdata['feat']
        output_predictions = model(blocks, input_feats)

        loss = loss_fn(output_predictions, output_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(output_labels[: 5])
            print(loss.item())


def train_graph(model, loss_fn, optimizer, graph, labels, train_idx, device):
    assert isinstance(model, torch.nn.Module)
    model.train()

    graph, labels = graph.to(device), labels.flatten().to(device)
    feat = graph.ndata['feat'].to(device)
    mask = torch.rand(train_idx.shape) < 0.5
    train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = loss_fn(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    optimizer.step()

    print(loss.item())

    return loss


if __name__ == "__main__":
    arxiv = Arxiv()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('trianing on {}'.format('cuda' if torch.cuda.is_available() else 'cpu'))

    model = MultiLayersGCN(
            arxiv.g.ndata['feat'].shape[-1], [256, 256], 40, 
            dropout=0.4, use_linear=True)
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', factor=0.5, patience=100, verbose=True, min_lr=1e-3)

    for _ in range(0, 500):
        loss = train_graph(model, loss_fn, optimizer, arxiv.g, arxiv.labels, arxiv.idx['train'], device)
        lr_scheduler.step(loss)



import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import dgl
from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler, NodeDataLoader
from model.data import Arxiv
from model.TwoLayersGCN import TwoLayersGCN
from model.TwoLayersSAGE import TwoLayersSAGE


def train(model, loss_fn, optimizer, dataloader, labels, device):
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


if __name__ == "__main__":
    arxiv = Arxiv()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('trianing on {}'.format('cuda' if torch.cuda.is_available() else 'cpu'))
    sampler = MultiLayerFullNeighborSampler(2)
    dataloader = NodeDataLoader(
            arxiv.g, arxiv.idx['train'], sampler, 
            batch_size=256, 
            shuffle=True, 
            num_workers=4)


    # model = TwoLayersGCN(arxiv.g.ndata['feat'].shape[-1], 128, 64, 40)
    model = TwoLayersSAGE(arxiv.g.ndata['feat'].shape[-1], 128, 64, 40)
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(0, 500):
        train(model, loss_fn, optimizer, dataloader, arxiv.labels, device)

 




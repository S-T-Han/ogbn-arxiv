import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import dgl
from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler, NodeDataLoader
from ogb.nodeproppred import Evaluator
from tqdm import tqdm

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


def train_graph(model, loss_fn, optimizer, graph, labels, train_idx):
    assert isinstance(model, torch.nn.Module)
    model.train()

    feat = graph.ndata['feat']
    mask = torch.rand(train_idx.shape) < 0.5
    train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = loss_fn(pred[train_pred_idx], labels[train_pred_idx].flatten())
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def eval_graph(model, evaluator, graph, labels, valid_idx, test_idx):
    assert isinstance(model, torch.nn.Module)
    assert isinstance(evaluator, Evaluator)
    model.eval()

    feat = graph.ndata['feat']
    pred = model(graph, feat)
    eval_acc = evaluator.eval({
        'y_pred': pred[valid_idx].argmax(dim=-1, keepdim=True),
        'y_true': labels[valid_idx]})['acc']
    test_acc = evaluator.eval({
        'y_pred': pred[test_idx].argmax(dim=-1, keepdim=True),
        'y_true': labels[test_idx]})['acc']

    return eval_acc, test_acc


def run_graph(
        model, loss_fn, optimizer, lr_scheduler, evaluator,
        graph, labels,
        train_idx, valid_idx, test_idx,
        epochs, device):
    assert isinstance(model, torch.nn.Module)
    model = model.to(device)
    graph, labels = graph.to(device), labels.to(device)

    max_eval_acc, final_test_acc = 0, 0
    for i in tqdm(range(0, epochs)):
        loss = train_graph(model, loss_fn, optimizer, graph, labels, train_idx)
        lr_scheduler.step(loss)
        eval_acc, test_acc = eval_graph(model, evaluator, graph, labels, valid_idx, test_idx)
        if eval_acc > max_eval_acc:
            max_eval_acc, final_test_acc = eval_acc, test_acc
            print('max eval acc: {}'.format(eval_acc))
            print('test acc: {}'.format(test_acc))

    print(max_eval_acc, final_test_acc)


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
    evaluator = Evaluator(name='ogbn-arxiv')

    run_graph(
            model, loss_fn, optimizer, lr_scheduler, evaluator, 
            arxiv.g, arxiv.labels,
            arxiv.idx['train'], arxiv.idx['valid'], arxiv.idx['test'], 
            epochs=3000, device=device)


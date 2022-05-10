import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset


# 载入数据集
dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='../../data/')
# dataloader

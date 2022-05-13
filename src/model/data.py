import os
from ogb.nodeproppred import DglNodePropPredDataset
import torch
import dgl


class Arxiv:
    def __init__(self):
        print('Loading Arxiv dataset...')
        self.data_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                    'data'))
        dataset = DglNodePropPredDataset(name='ogbn-arxiv', root=self.data_path)
        self.g, self.labels = dataset[0]
        self.g.add_edges(*self.g.all_edges()[::-1])
        self.g = self.g.remove_self_loop().add_self_loop()
        self.idx = dataset.get_idx_split()


if __name__ == "__main__":
    arxiv = Arxiv()
    print(arxiv.g.ndata['feat'].shape[-1])
    graph = arxiv.g
    assert isinstance(graph, dgl.DGLGraph)
    src, dst = graph.all_edges()
    print(src[: 5])
    print(dst[: 5])
    print(graph.has_edges_between(torch.tensor([104447]), torch.tensor([13091])))
    print(graph.has_edges_between(torch.tensor([13091]), torch.tensor([104447])))
    print(graph.has_edges_between(torch.tensor([104447]), torch.tensor([104447])))




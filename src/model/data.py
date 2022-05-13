import os
from ogb.nodeproppred import DglNodePropPredDataset
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
        self.g.remove_self_loop().add_self_loop()
        self.idx = dataset.get_idx_split()


if __name__ == "__main__":
    arxiv = Arxiv()
    print(type(arxiv.labels))
    print(arxiv.labels[: 20])
    print(arxiv.g.ndata['feat'].shape[-1])



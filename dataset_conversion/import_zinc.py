
import networkx as nx
import pickle
from os.path import join
from os import makedirs
import numpy as np
import json
import torch_geometric as pyg


def convert_from_pyg(save_path:str='data/graphdbs/'):

    graphs = list()
    labels = list()
    metas = list()
    global_idx = 0

    makedirs(save_path, exist_ok=True)

    for split in ['train', 'val', 'test']:

        dataset_name = 'ZINC_subset'
        dataset = pyg.datasets.ZINC(root='pygdata', subset=True, split=split)

        for i, g in enumerate(dataset):
            nxg = pyg.utils.to_networkx(g, to_undirected=True)
            graphs.append(nxg)
            
            meta = {'vertices': len(nxg.nodes), 
                    'edges': len(nxg.edges),
                    'split': split,
                    'idx_in_split': i, 
                    'idx': global_idx}
            metas.append(meta)

            labels.append(g.y)

            global_idx += 1

    with open(join(save_path, dataset_name.upper() + '.meta'), 'w') as f:
        json.dump(metas, f)    

    with open(join(save_path, dataset_name.upper() + '.graph'), 'wb') as f:
        pickle.dump(graphs, f)

    # TODO: we don't store any labels, yet.
    with open(join(save_path, dataset_name.upper()) + '.y', 'wb') as f:
        pickle.dump(np.zeros(len(graphs)), f)

if __name__ == '__main__':
    convert_from_pyg()


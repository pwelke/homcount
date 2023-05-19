import networkx as nx
import pickle
from os.path import join
import numpy as np
import json
import torch_geometric as pyg
import torch


def convert_from_pyg(dataset_name, save_path:str='data/graphdbs'):

    graphs = list()
    labels = list()
    metas = list()
    global_idx = 0

    for split in ['full']:

        dataset = pyg.datasets.TUDataset(name=dataset_name, root='pygdata')

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

    with open(join(save_path, dataset_name.upper()) + '.y', 'wb') as f:
        labels = torch.tensor(labels).detach().numpy()
        pickle.dump(labels, f)


if __name__ == '__main__':
    datasets = ['MUTAG', 'BZR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'NCI1', 'ENZYMES', 'DD', 'COLLAB']

    for d in datasets:
        convert_from_pyg(d)


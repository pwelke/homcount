## TODO: currently requires a different environment with ogb installed

from ogb.graphproppred import GraphPropPredDataset
import networkx as nx
import pickle
from os.path import join
import numpy as np
import json


def convert_from_ogb(dataset_name, save_path:str='graph-homomorphism-network/data/'):
    
    dataset = GraphPropPredDataset(name = dataset_name, root = 'ogbdata/')

    # TODO: we don't use any train test validation split, yet
    split_idx = dataset.get_idx_split() 


    graphs = list()
    labels = list()
    metas = list()

    for g in dataset.graphs:
        graph = nx.Graph() 
        graph.add_nodes_from(range(g['num_nodes']))
        graph.add_edges_from(zip(g['edge_index'][0], g['edge_index'][1]))
        graphs.append(graph)
        meta = {'vertices': g['num_nodes'], 'edges': g['edge_index'][0].shape[0]}
        metas.append(meta)

    for split in ['train', 'valid', 'test']:
        for i, x in enumerate(split_idx[split]):
            metas[x]['split'] = split
            metas[x]['idx_in_split'] = i
            metas[x]['idx'] = int(x)

    with open(join(save_path, dataset_name.upper() + '.meta'), 'w') as f:
        json.dump(metas, f)    

    with open(join(save_path, dataset_name.upper() + '.graph'), 'wb') as f:
        pickle.dump(graphs, f)

    # TODO: we don't store any labels, yet.
    with open(join(save_path, dataset_name.upper()) + '.y', 'wb') as f:
        pickle.dump(np.zeros(len(graphs)), f)

datasets = ['ogbg-moltox21',
            'ogbg-molesol',
            'ogbg-molbace',
            'ogbg-molclintox',
            'ogbg-molbbbp',
            'ogbg-molsider',
            'ogbg-moltoxcast',
            'ogbg-mollipo',
            'ogbg-molhiv',]

for dataset_name in datasets:
    convert_from_ogb(dataset_name)


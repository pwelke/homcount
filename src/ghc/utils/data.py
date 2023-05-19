from collections import defaultdict
import pickle as pkl
import networkx as nx
import numpy as np
import random
import os
import random
import json
from itertools import repeat
from sklearn.model_selection import KFold


def to_onehot(y, nmax=None):
    '''Convert a 1d numpy array to 2d one hot.'''
    if y.size == 0:
        return y
    if nmax is None:
        nmax = y.max()+1
    oh = np.zeros((y.size, nmax))
    oh[np.arange(y.size), y] = 1
    return oh

def from_onehot(y):
    features = np.array([np.where(z == 1) for z in y])
    return features.reshape([-1,1])

def save_precompute(X, dataset, hom_type, hom_size, pattern_count, run_id, dloc):
    dataf = os.path.abspath(dloc)
    tmp_str = f"{dataf}/{dataset}_{hom_type}_{hom_size}_{pattern_count}_{run_id}.hom"
    with open(tmp_str, 'wb') as f:
        pkl.dump(X, f)

def precompute_patterns_file_handle(dataset, hom_type, hom_size, pattern_count, run_id, dloc):
    dataf = os.path.abspath(dloc)
    tmp_str = f"{dataf}/{dataset}_{hom_type}_{hom_size}_{pattern_count}_{run_id}.patterns"
    return open(tmp_str, 'wb')


def load_data_for_json(fname, dloc):
    graphs, feats, y = load_data(fname, dloc)
    name = os.path.abspath(os.path.join(dloc, fname + '.meta'))
    with open(name, 'r') as f:
        metas = json.load(f)
    return graphs, feats, y, metas

def hom2json(metas, homX, ys):
    for meta, hom, y in zip(metas, homX, ys):
        meta['counts'] = hom.tolist()
        meta['y'] = y
    return metas

def save_json(meta, dataset, hom_type, hom_size, pattern_count, run_id, dloc, suffix='homson'):
    dataf = os.path.abspath(dloc)
    tmp_str = f"{dataf}/{dataset}_{hom_type}_{hom_size}_{pattern_count}_{run_id}.{suffix}"
    with open(tmp_str, 'w') as f:
        json.dump(meta, f)

def load_json(dataset, hom_type, hom_size, pattern_count, run_id, dloc, suffix='homson'):
    dataf = os.path.abspath(dloc)
    tmp_str = f"{dataf}/{dataset}_{hom_type}_{hom_size}_{pattern_count}_{run_id}.{suffix}"
    with open(tmp_str, 'r') as f:
        X = json.load(f)
    return X

def load_precompute(dataset, hom_type, hom_size, pattern_count, run_id, dloc):
    dataf = os.path.abspath(dloc)
    tmp_str = f"{dataf}/{dataset}_{hom_type}_{hom_size}_{pattern_count}_{run_id}.hom"
    with open(tmp_str, 'rb') as f:
        X = pkl.load(f)
    return X

def load_precompute_patterns(dataset, hom_type, hom_size, pattern_count, run_id, dloc):
    dataf = os.path.abspath(dloc)
    tmp_str = f"{dataf}/{dataset}_{hom_type}_{hom_size}_{pattern_count}_{run_id}.patterns"
    with open(tmp_str, 'rb') as f:
        X = pkl.load(f)
    return X

def load_data(dname, dloc):
    """Load datasets"""
    X = None
    y = None
    graphs = None
    name = os.path.abspath(os.path.join(dloc, dname))
    with open(name+".graph", "rb") as f:
        graphs = pkl.load(f)
    with open(name+".y", "rb") as f:
        y = pkl.load(f)
    if os.path.exists(name+".X"):
        with open(name+".X", "rb") as f:
            X = pkl.load(f)
    else:
        X = [np.ones(g.number_of_nodes()).reshape([-1,1]) for g in graphs]
    return graphs, X, y


def load_folds(dname, dloc):
    """Load preassigned 10-folds splits for each datasets"""
    splits = None
    name = os.path.abspath(os.path.join(dloc, dname))
    with open(name+".folds", "rb") as f:
        splits = pkl.load(f)
    return splits


def create_folds(dname, dloc, X):
    """Create 10-fold splits for a dataset"""

    folder = KFold(n_splits=10, shuffle=True)
    splits = [s for s in folder.split(X)]

    name = os.path.abspath(os.path.join(dloc, dname))
    with open(name+".folds", "wb") as f:
        pkl.dump(splits, file=f)
    return splits

def drop_nodes(graph, x, rate=1):
    #### Remove nodes
    n = graph.number_of_nodes()
    ng = graph.copy()
    if type(rate) is float:
        num_drop = int(rate*n)
    else:
        num_drop = rate
    droplist = np.random.choice(ng.nodes(), size=num_drop, replace=False)
    ng.remove_nodes_from(droplist)
    #### Reindex to consecutive integers
    mapping = dict([(i, j) for j, i in enumerate(ng.nodes())])
    ng = nx.relabel_nodes(ng, mapping=mapping)
    newx = x[list(mapping.keys()), :]
    return ng, newx

def augment_data(graphs, X, y, samples_per_graph, rate=1):
    new_graphs = []
    new_X = []
    new_y = [[ny] for dupy in y for ny in repeat(dupy, samples_per_graph)]
    for g, x in zip(graphs, X):
        gen_data = [drop_nodes(g, x, rate) for _ in range(samples_per_graph)]
        gen_graphs, gen_x = zip(*gen_data)
        new_graphs.extend(gen_graphs)
        new_X.extend(gen_x)
    return new_graphs, np.array(new_X, dtype=object), np.array(new_y)

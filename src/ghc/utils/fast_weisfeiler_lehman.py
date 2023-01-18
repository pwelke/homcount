import sys 
sys.path.append('graph-homomorphism-network/src/')


import scipy.sparse as sparse
import numpy as np
import networkx as nx
from ghc.utils.data import from_onehot, to_onehot

import os, inspect

module_dir_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # get path to the directory of current module
logprimefile = os.path.join(module_dir_path,'logprimes1.npy')
print(logprimefile)
# path stored in useful_file_path. do whatever you want!

primes = np.load(logprimefile)

def compress(labels: np.array):
    '''Weisfeiler Leman Label compression'''
    _, inv = np.unique(labels, return_inverse=True)
    # use the indices to uniq array to select first |uniq| primes
    labels = primes[inv]
    return labels

def compress_int(labels: np.array):
    '''final step transform to int'''
    _, inv = np.unique(labels, return_inverse=True)
    return inv

def homsub_format_wl_nodelabels(graphs, vertex_features, n_iter):
    return homsub_format_wl_nodelabels_nx(graphs, vertex_features, n_iter)

def homsub_format_wl_nodelabels_nx(graphs, vertex_features, n_iter):
    
    G = nx.disjoint_union_all(graphs)
    
    if vertex_features is not None:
        v = np.vstack(vertex_features)
        vv = from_onehot(v)
    else:
        vv = None

    adj = nx.to_scipy_sparse_matrix(G)

    wl_labels = wl_direct_scipysparse(adj, vertex_labels=vv, n_iter=n_iter)

    oh = to_onehot(compress_int(wl_labels))

    wl_features = list()
    i = 0
    for g in graphs:
        n = g.number_of_nodes()
        wl_features.append(oh[i:i+n, :])
        i += n

    return wl_features

def homsub_format_wl_nodelabels_scipy(graphs, vertex_features, n_iter):
    
    adj = sparse.block_diag([nx.to_scipy_sparse_matrix(g) for g in graphs], format='csr')
    
    if vertex_features is not None:
        v = np.vstack(vertex_features)
        vv = from_onehot(v)
    else:
        vv = None

    # adj = nx.to_scipy_sparse_matrix(G)

    wl_labels = wl_direct_scipysparse(adj, vertex_labels=vv, n_iter=n_iter)

    oh = to_onehot(compress_int(wl_labels))

    wl_features = list()
    i = 0
    for g in graphs:
        n = g.number_of_nodes()
        wl_features.append(oh[i:i+n, :])
        i += n

    return wl_features


def wl_direct_scipysparse(a: sparse.csr_matrix, vertex_labels=None,  n_iter=5):
    if vertex_labels is None:
        oldlbl = primes[0] * np.ones(a.shape[0])
    else:
        oldlbl = vertex_labels

    newlbl = oldlbl # only relevant for n_iter=0

    for i in range(n_iter):
        newlbl = np.pi * oldlbl + a @ oldlbl

        if i < n_iter-1:
            oldlbl = compress(newlbl)

    return newlbl


def compare_equivalence_classes(hom_features, wl_features):
    '''returns the difference between number of unique rows in first argument and 
    number of unique rows in second argument.
    
    That is, the return is positive, if first argument has 'more expressive power'
    than second argument'''

    hom_uniq = np.unique(hom_features, axis=0)
    wl_uniq = np.unique(wl_features, axis=0)
    diff = hom_uniq.shape[0] - wl_uniq.shape[0]
    print(f'HINT {diff}')
    # not yet really what we want, but simple
    return diff


def test_homsub_format_wl_nodelabels(n_patterns=10, n_iter=2):
    import time
    from ghc.generate_k_tree import get_small_patterns, get_pattern_list

    ps, _ = get_pattern_list(30, n_patterns)

    tic = time.time()
    a = homsub_format_wl_nodelabels_nx(ps, None, n_iter)
    print(f'for n_patterns={n_patterns} nx took {time.time() - tic}')
    
    tic = time.time()
    b = homsub_format_wl_nodelabels_scipy(ps, None, n_iter)
    print(f'for n_patterns={n_patterns} sp took {time.time() - tic}')

    for g, h in zip(a, b):
        if not np.array_equal(g,h):
            print('issue!')
            break

if __name__ == '__main__':
    for i in range(10):
        test_homsub_format_wl_nodelabels(n_patterns=(i+1)*100)

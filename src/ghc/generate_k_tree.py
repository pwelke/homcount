import networkx as nx
import random
import itertools

from ghc.utils.HomSubio import HomSub, PACE_graph_format
import numpy as np


def random_ktree_decomposition(N, k, seed=None):
    '''Sample a random ktree on N vertices.
    
    To this end, draw a random tree that defines the topology of the 
    tree decomposition and then randomly create bags for a full, nice 
    tree decomposition.
    '''

    if k > N - 1:
        raise ValueError(f'k(={k})+1 cannot be larger than N(={N})')

    # sample a random tree for the tree decomposition
    T = nx.generators.random_tree(N-k, seed=seed)
    bfs = nx.bfs_edges(T, 0)

    rnd = random.seed(seed)

    # construct bags of a k-tree. The first bag consists of the first k+1 vertices and is fully connected
    bags = [None for _ in range(N-k)]
    bags[0] = [v for v in range(k+1)] # I assume shuffling is not necessary, as T is random
    edges = [e for e in itertools.combinations(bags[0], 2)]

    candidates = [v for v in range(k+1, N)]

    PACE_tdstring = f's td {N-k} {k+1} {N}\n'
    PACE_bagstrings = ['' for _ in range(N-k)]
    PACE_bagstrings[0] = 'b 1 ' + ' '.join([str(v + 1) for v in bags[0]])

    PACE_tdedges = ''

    for i, e in enumerate(bfs):
        bag = bags[e[0]].copy()
        deleted_vertex = bag.pop(random.randint(0, k-1))
        new_vertex = candidates[i]
        for v in bag:
            edges.append((v, new_vertex))
        bag.append(new_vertex)
        bags[e[1]] = bag

        PACE_bagstrings[e[1]] = f'b {e[1] + 1} ' + ' '.join([str(v + 1) for v in bag]) 
        PACE_tdedges += f'{min(e[0], e[1]) + 1} {max(e[0], e[1]) + 1}\n'

    PACE_tdstring += '\n'.join(PACE_bagstrings) + '\n' + PACE_tdedges    
    tree_decomposition = (T, bags)

    return edges, tree_decomposition, PACE_tdstring


def erdos_filter(edges, p=0.9, seed=None):
    '''Delete edges from edge list i.i.d. with probability 1-p.
    I.e., keep any edge with probability p'''

    random.seed(seed)

    filtered_edges = list()
    for e in edges:
        if random.random() < p:
            filtered_edges.append(e)

    return filtered_edges


def connected_filter(P):
    '''Return the list of connected components, as networkx graphs'''

    S = [P.subgraph(c).copy() for c in nx.connected_components(P)]
    return S


def partial_ktree_sample(N, k, p, seed=None):
    '''Returns a list of networkx graphs that are the connected components of a 
    partial ktree that was obtained by deleting edges with probability p from a 
    random k tree on N vertices. '''

    edges, td, string = random_ktree_decomposition(N, k, seed=seed)
    filtered_edges = erdos_filter(edges, p=p, seed=seed)
    filtered_graph = nx.empty_graph(n=N)
    filtered_graph.add_edges_from(filtered_edges)
    # connected_components = connected_filter(filtered_graph)

    return filtered_graph, string


def Nk_strategy_geom(max_size, pattern_count, p='by_max'):

    if p == 'by_max':
        p = 1. - 1. / max_size

    # draw sizes from uniform distribution
    sizes = np.random.randint(2, max_size+1, size=pattern_count)

    # draw treewidths from geometric distribution, but bounded by size - 1
    treewidths = np.random.default_rng().geometric(p=p, size=pattern_count)
    treewidths = np.where(treewidths<sizes-1, treewidths, sizes - 1)

    return sizes, treewidths


def Nk_strategy_poisson(max_size, pattern_count, lam='by_max'):

    if lam == 'by_max':
        lam = (1. + 3 * np.log(max_size)) / max_size

    # draw sizes from uniform distribution
    sizes = np.random.randint(2, max_size+1, size=pattern_count)

    # draw treewidths from geometric distribution, but bounded by size - 1
    treewidths = 1 + np.random.default_rng().poisson(lam=lam, size=pattern_count)
    treewidths = np.where(treewidths<sizes-1, treewidths, sizes - 1)

    return sizes, treewidths


def Nk_strategy_fiddly(max_size, pattern_count, lam='by_max'):

    if lam == 'by_max':
        lam = (1. + np.log(max_size)) / max_size

    # draw sizes from uniform distribution
    sizes = np.random.default_rng().geometric(p=0.1, size=pattern_count) #randint(2, max_size+1, size=pattern_count)

    # draw treewidths from geometric distribution, but bounded by size - 1
    treewidths = np.random.randint(1, 4, size=pattern_count) + np.random.default_rng().poisson(lam=lam, size=pattern_count)
    treewidths = np.where(treewidths<sizes-1, treewidths, sizes - 1)

    return sizes, treewidths


# this is currently our default selection strategy for pattern sizes and treewidths
Nk_strategy = Nk_strategy_fiddly


def min_embedding(pattern_list, graph_list, td_list):
    '''For each (transaction) graph, we use only those patterns that
    have smaller or equal size. This implements the min-kernel as 
    descibed in the paper. 

    TODO: We could speed up homomorphism counting by counting 
    only those patterns in a graph that have smaller or equal size. 
    '''

    pattern_sizes = np.array([len(g.nodes) for g in pattern_list]).reshape([1, len(pattern_list)])
    graph_sizes = np.array([len(g.nodes) for g in graph_list]).reshape([len(graph_list), 1])

    full_embeddings = HomSub(pattern_list, graph_list, td_list=td_list)

    # note that this is numpy broadcast magic given the specific shapes of pattern_sizes and graph_sizes
    min_embeddings = np.where(pattern_sizes<=graph_sizes, full_embeddings, 0)

    return min_embeddings


def get_pattern_list(size, pattern_count):
    
    # tw_downweighting_p = 0.35
    partial_ktree_edge_keeping_p = 0.9
    
    # TODO: handling possibly disconnected patterns, now. 
    # this function can be simplified
    kt_list = list()
    td_list = list()
    while len(kt_list) < pattern_count:
        
        sizes, treewidths = Nk_strategy(size, 1, 'by_max')
        pattern, td = partial_ktree_sample(N=sizes[0], k=treewidths[0], p=partial_ktree_edge_keeping_p)

        kt_list += [pattern]
        td_list += [td]
        
    kt_list = kt_list[:pattern_count] # the above might result in more than pattern_count patterns
    td_list = td_list[:pattern_count]
    return kt_list, td_list

def random_ktree_profile(graphs, size='max', density=False, seed=8, pattern_count=50, **kwargs):

    if size == 'max':
        size = max([len(g.nodes) for g in graphs])

    kt_list, td_list = get_pattern_list(size, pattern_count)
    # homX = list()
    # for G in graphs:
    #     homX += [hom(G, kt, density=density) for kt in kt_list]
    # return homX

    return min_embedding(pattern_list=kt_list, graph_list=graphs, td_list=td_list)


if __name__ == '__main__':


    pattern_list = [nx.path_graph(i) for i in range(2,5)]
    graph_list = [nx.path_graph(i) for i in range(2,10)]

    patterns, tds = get_pattern_list(10, 2)

    print(f'{PACE_graph_format(patterns[0])}')
    print(tds[0])

    arr = HomSub(patterns, graph_list, tds)
    print(arr)

    # print(Nk_strategy_geom(30, 20, p='by_max'))
    # print(Nk_strategy_poisson(30,20))
    # print(Nk_strategy_fiddly(30,20))
    
    # print(random_ktree_profile(graph_list, pattern_count=10))



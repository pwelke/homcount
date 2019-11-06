from collections import Counter
import pickle as pkl
import networkx as nx
import numpy as np
import random
import torch
import os

from sklearn.model_selection import StratifiedKFold

try:
    from graph_tool.all import Graph as gtGraph
except:
    print("Please install graph-tool to run subgraph density. "\
          "Only pre-computed vectors are available without graph-tool")

try:
    from homlib import Graph as hlGraph
except:
    print("Please install homlib graph library for fast tree homomorphism.")


def to_onehot(X):
    """Convert a 1d numpy array to 2d one hot."""
    oh = np.zeros((X.size, X.max()+1)) 
    oh[np.arange(X.size), X] = 1
    return oh


def save_precompute(X, dataset, hom_type, hom_size):
    dataf = os.path.dirname(os.path.abspath(__file__))+"/data"
    tmp_str = "{}/{}/{}_{}_{}.pkl"
    with open(tmp_str.format(dataf,dataset,dataset,hom_type,hom_size), 
              'wb') as f:
        pkl.dump(X, f)


def load_precompute(dataset, hom_type, hom_size):
    dataf = os.path.dirname(os.path.abspath(__file__))+"/data"
    tmp_str = "{}/{}/{}_{}_{}.pkl"
    try:
        with open(tmp_str.format(dataf,dataset,dataset,hom_type,hom_size), 
                  'rb') as f:
            X = pkl.load(f)
    except:
        X = []
    return X
    

def nx2gt(nxg):
    """Simple function to convert s2v to graph-tool graph."""
    gt = gtGraph(directed=nxg.is_directed())
    gt.add_edge_list(nxg.edges())
    return gt


def nx2homg(nxg):
    """Convert nx graph to homlib graph format. Only 
    undirected graphs are supported. 
    originally suggested by Takanori Maehara (@spagetti-source)"""
    n = nxg.number_of_nodes()
    G = hlGraph(n)
    for (u, v) in nxg.edges():
        G.addEdge(u,v)
    return G


def tree_list(size=6, to_homlib=False):
    """Generate nonisomorphic trees up to size `size`."""
    t_list = [tree for i in range(2,size+1) for tree in \
                       nx.generators.nonisomorphic_trees(i)]
    if to_homlib:
        t_list = [nx2homg(t) for t in t_list]
    return t_list


def cycle_list(size=6, to_homlib=False):
    """Generate undirected cycles up to size `size`. Parallel
    edges are not allowed."""
    c_list = [nx.generators.cycle_graph(i) for i in range(2,size+1)]
    if to_homlib:
        c_list = [nx2homg(c) for c in c_list]
    return c_list


def path_list(size=6, to_homlib=False):
    """Generate undirected paths up to size `size`. Parallel
    edges are not allowed."""
    p_list = [nx.generators.path_graph(i) for i in range(2,size+1)]
    if to_homlib:
        p_list = [nx2homg(p) for p in p_list]
    return p_list


def graph_type(g):
    if g.__module__ == 'homlib':
        return 'hl'
    elif g.__module__ == 'networkx.classes.graph':
        return 'nx'
    else:
        raise TypeError("Unsupported graph type: {}".format(str(g)))
    #TODO(N): Add for graph-tool type if needed.


def load_tud_data(dset, combine_tag_feat=False):
    """Utility function to load other datasets for graph
    classification besides the one provided by GIN. These graphs
    follow TU Dormund's format.
    https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

    combine_tag_feat: append node tags to node features (stored in features).
    """
    dataf = os.path.dirname(os.path.abspath(__file__))+"/data"
    adj_file = "{}/{}/{}_A.txt".format(dataf, dset, dset)
    indicator_file = "{}/{}/{}_graph_indicator.txt".format(dataf, dset, dset)
    graph_label_file = "{}/{}/{}_graph_labels.txt".format(dataf, dset, dset)
    node_label_file = "{}/{}/{}_node_labels.txt".format(dataf, dset, dset)
    node_attr_file = "{}/{}/{}_node_attributes.txt".format(dataf, dset, dset)

    # Read and convert node tags if available (i.e. node labels)
    node_label_by_index = []
    if os.path.exists(node_label_file):
        node_label_to_index = dict()
        next_index = 0
        with open(node_label_file, 'r') as nlf:
            for dline in nlf.readlines():
                raw_node_label = dline.strip()
                if raw_node_label not in node_label_to_index:
                    node_label_to_index[raw_node_label] = next_index
                    next_index += 1         
                indexed_node_label = node_label_to_index[raw_node_label]
                node_label_by_index.append(indexed_node_label)
        del node_label_to_index
        node_label_by_index = np.array(node_label_by_index, dtype=int)
        node_label_by_index = to_onehot(node_label_by_index)

    # Load node attribute (i.e. node features) if available
    node_attr = []
    if os.path.exists(node_attr_file):
        node_attr = np.loadtxt(node_attr_file, delimiter=',')

    # Read graph labels
    graph_label = np.loadtxt(graph_label_file)
    unique_label = np.unique(graph_label)
    nclass = len(unique_label)
    ngraph = len(graph_label)
    raw_label_to_id = {unique_label[i]: i for i in range(nclass)}
    graph_label = [raw_label_to_id[rl] for rl in graph_label]

    # Read indicator file to know number of lines to read
    graph_indicator = np.loadtxt(indicator_file, dtype=int)
    raw_graph_id_to_num_edge = Counter(graph_indicator)
    assert len(raw_graph_id_to_num_edge) == ngraph, "Inconsistent num graphs!"

    # Read graph structure
    g_list = []
    raw_edge_list = np.loadtxt(adj_file, delimiter=',', dtype=int)
    offset = 0
    for i, (key, value) in enumerate(raw_graph_id_to_num_edge.items()):
        nxg = nx.from_edgelist(raw_edge_list[offset:offset+value])
        gl = graph_label[i]
        if len(node_label_by_index) != 0:
            lookup_indices = [i-1 for i in nxg.nodes()]
            nt = node_label_by_index[lookup_indices]
        else:
            nt = None
        if len(node_attr) != 0:
            lookup_indices = [i-1 for i in nxg.nodes()]
            na = node_attr[lookup_indices]
        else:
            na = None
        # Append tags to features if requested
        if combine_tag_feat:
            if na is None: 
                na = nt
            elif nt is not None:
                na = np.concatenate((na, nt), axis=1)
        g = S2VGraph(nxg, gl, node_tags=nt, node_features=na)
        offset += value
        g_list.append(g)
    return g_list, nclass

##############################################################
### Copied from https://github.com/weihua916/powerful-gnns ###
##############################################################
def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    dataf = os.path.dirname(os.path.abspath(__file__))+"/data"
    with open('{}/{}/{}.txt'.format(dataf,dataset,dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]],\
                                np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)),\
                        [tag2index[tag] for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list
##############################################################
##############################################################
##############################################################

# Taken from GIN code but changed a bit to allow node features
class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.node_features = node_features

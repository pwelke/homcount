import networkx as nx
from networkx.relabel import convert_node_labels_to_integers
import os
import numpy as np
from tqdm import tqdm

import sys
import subprocess
import tempfile
import re



def HomSub(pattern_list, graph_list):
    '''Compute homomorphism counts for a batch of patterns and a batch of 
    (transaction) graphs using HomSub. For each pattern-transaction pair, we call HomSub
    anew.

    Files which are used to communicate data between Python and HomSub
    are written to a temp folder. '''

    # output_directory = tempfile.mkdtemp()
    graph_directory = tempfile.mkdtemp()
        
    # store patterns and graphs
    write_PACE_graphs(graph_list, folder=graph_directory, prefix='graph')
    write_PACE_graphs(pattern_list, folder=graph_directory, prefix='pattern')

    # invoke DISC and collect issues
    cwd = './'

    # files = [f for f in filter(lambda x: x.endswith('.txt'), os.listdir(graph_directory))]
    # files.sort(key=lambda f: int(re.sub('\D', '', f)))
    

    ngraphs = len(graph_list)
    npatterns = len(pattern_list)
    with open(os.path.join(graph_directory, 'features.csv'), 'w') as features:
        for ig in range(ngraphs):
            for jp in range(npatterns):
                sys.stderr.write(f'pattern_{jp} n={len(pattern_list[jp].nodes)} m={len(pattern_list[jp].edges)}, graph_{ig} n={len(graph_list[ig].nodes)} m={len(graph_list[ig].edges)}' + '\n')
                args = ['./HomSub/experiments-build/experiments/experiments',
                        '-count-hom',
                        '-h', os.path.join(graph_directory, f'pattern_{jp}.gr'), 
                        '-g', os.path.join(graph_directory, f'graph_{ig}.gr')]
                try:
                    report = subprocess.run(args, cwd=cwd, stdout=features, stderr=sys.stderr, text=True, check=True)
                    features.flush()
                except subprocess.CalledProcessError as e:
                    sys.stderr.write(f'{e}')
                    features.write('-1\n')


    hom_counts = np.loadtxt(os.path.join(graph_directory, 'features.csv')).reshape([ngraphs, npatterns])

    # return everything
    return hom_counts


def PACE_graph_format(g):
    string = f'p tw {len(g.nodes)} {len(g.edges)}\n'
    string += '\n'.join([f'{e[0] + 1} {e[1] + 1}' for e in g.edges])
    string += '\n'
    return string


def write_PACE_graphs(graphs, folder, prefix):
    for i, g in enumerate(graphs):
        string = PACE_graph_format(g)
        with open(os.path.join(folder, f'{prefix}_{i}.gr'), 'w') as f:
            f.write(string)


if __name__ == '__main__':

    # create_fixed_pattern_set()

    pattern_list = [nx.path_graph(i) for i in range(2,5)]
    graph_list = [nx.path_graph(i) for i in range(2,10)]

    # print(PACE_graph_format(graph_list[2]))

    HomSub(pattern_list, graph_list)




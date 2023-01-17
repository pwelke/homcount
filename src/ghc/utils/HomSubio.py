import networkx as nx
from networkx.relabel import convert_node_labels_to_integers
import os
import numpy as np
from tqdm import tqdm

import sys
import subprocess
import tempfile
import re



def HomSub(pattern_list, graph_list, td_list, verbose=False, min_embedding=False):
    '''Compute homomorphism counts for a batch of patterns and a batch of 
    (transaction) graphs using HomSub. For each pattern-transaction pair selected for 
    computation, we call HomSub anew.

    min_embedding: If True, for each (transaction) graph, we use only those patterns that
        have smaller or equal size. This implements the min-kernel as 
        descibed in the paper. 
        If False, for each (transaction) graph, we compute the hom counts for all 
        patterns. This implements a standard kernel which is likely useful 
        for feeding into an MLP

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
        for ig in tqdm(range(ngraphs)):
            for jp in range(npatterns):

                # HomSub expects a tree decomposition of the pattern in a file named tam.out
                with open('tam.out', 'w') as td_file:
                    td_file.write(td_list[jp])

                if verbose:
                    sys.stderr.write(f'pattern_{jp} n={len(pattern_list[jp].nodes)} m={len(pattern_list[jp].edges)}, graph_{ig} n={len(graph_list[ig].nodes)} m={len(graph_list[ig].edges)}' + '\n')
                
                args = ['./HomSub/experiments-build/experiments/experiments',
                        '-count-hom',
                        '-h', os.path.join(graph_directory, f'pattern_{jp}.gr'), 
                        '-g', os.path.join(graph_directory, f'graph_{ig}.gr')]
                try:
                    if min_embedding:
                        # for the min_embedding, we don't need to compute a count for patterns larger than the current graph 
                        if len(graph_list[ig].nodes) >= len(pattern_list[jp].nodes):  
                            if verbose:
                                sys.stderr.write('doing it\n')             
                            report = subprocess.run(args, cwd=cwd, stdout=features, stderr=sys.stderr, text=True, check=True)
                        else:
                            if verbose:
                                sys.stderr.write('skipping it\n')                            
                            features.write('0\n')
                    else:
                        report = subprocess.run(args, cwd=cwd, stdout=features, stderr=sys.stderr, text=True, check=True)
                    features.flush()
                except subprocess.CalledProcessError as e:
                    sys.stderr.write(f'{e}')
                    features.write('-1\n')

    # homcounts get large. HomSub uses long long int, 
    # but it's unclear how large this is on any given system. 
    # so we use int128, as we use this lateron anyways.
    # note that hom_counts might still contain overflowed values from HomSub
    # filter at your own expense.
    hom_counts = np.loadtxt(os.path.join(graph_directory, 'features.csv'), dtype=np.int64).reshape([ngraphs, npatterns])

    # return everything
    return hom_counts
    

def PACE_graph_format(g):
    string = f'p tw {len(g.nodes)} {len(g.edges)}\n'
    string += '\n'.join([f'{min(e[0], e[1]) + 1} {max(e[0], e[1]) + 1}' for e in g.edges])
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

    HomSub(pattern_list, graph_list, None)




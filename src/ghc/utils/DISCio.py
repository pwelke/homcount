import networkx as nx
from networkx.relabel import convert_node_labels_to_integers
import os

import subprocess
import tempfile

def id_to_str(n):
    digits = [int(d) for d in str(n)]

    letter_string = ""
    for d in digits:
        letter_string += chr(d+ord('A'))

    return letter_string


def networkxToDISCPatternBatch(graphs, file):
    '''expects open TextIOWrapper that can be written to'''

    for i, g in enumerate(graphs):
        if i>0:
            s = "\n"
        else: 
            s = ""

        g = convert_node_labels_to_integers(g)
        s += "t"+str(i)+" "
        at_least_one_edge = False
        for v in g.nodes:
            for w in g.neighbors(v):
                if int(str(v)) < int(str(w)):
                    s += id_to_str(int(str(v)))+"-"+id_to_str(int(str(w)))+";"
                    at_least_one_edge = True

        # s = s[:-1] #remove last ";". but do we need this??? yes we do!

        if at_least_one_edge:
            file.write(s)
        
    return


def networkxToDISCDataGraphBatch(graphs, dir):

    for i, g in enumerate(graphs):

        g = convert_node_labels_to_integers(g, first_label=1) #DISC seems to expect 1,..,n labels and not 0,..,n-1

        with open(os.path.join(dir, "graph" + str(i) + ".txt"), 'w') as file:
            for v in g.nodes:
                for w in g.neighbors(v):
                    if int(str(v)) < int(str(w)):
                        file.write(str(v)+" "+str(w)+"\n")

def readDISCcounts(dir):
    pass


def DISChom(pattern_list, graph_list):
    '''Compute homomorphism counts for a batch of patterns and a batch of 
    (transaction) graphs using DISC.

    Files which are used to communicate data between Python and DISC (hadoop) 
    are written to a temp folder and deleted afterwards. '''

    # open temp directory and pattern file
    with tempfile.TemporaryDirectory() as tmp_directory:
        with tempfile.TemporaryDirectory() as graph_directory:
        
            # store patterns and graphs
            pattern_file = os.path.join(tmp_directory, 'patterns')
            with open(pattern_file, 'w') as pattern_fileobj:
                networkxToDISCPatternBatch(pattern_list, pattern_fileobj)
            networkxToDISCDataGraphBatch(graph_list, graph_directory)

            # invoke DISC and collect issues
            cwd = '../DISC/'
            args = ['sbt', f'runMain org.apache.spark.disc.BatchBatchSubgraphCounting --queryType HOM --executionMode Result --environment Local --queryFile {pattern_file} -t {graph_directory} --output {os.path.join(tmp_directory, "homcount")}']
            report = subprocess.run(args, cwd=cwd, capture_output=True, check=True)

            # collect hom counts
            hom_counts = 1(dir)

            # return everything
            return hom_counts


if __name__ == '__main__':
    pattern_list = [nx.path_graph(i) for i in range(2,5)]
    graph_list = [nx.path_graph(i) for i in range(2,10)]

    DISChom(pattern_list, graph_list)





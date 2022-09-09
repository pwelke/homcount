import networkx as nx
from networkx.relabel import convert_node_labels_to_integers
def id_to_str(n):
    digits = [int(d) for d in str(n)]

    letter_string = ""
    for d in digits:
        letter_string += chr(d+ord('A'))

    return letter_string

def networkxToDISCPatternBatch(graphs, filePath):
    with open(filePath, 'w') as file:

        for i, g in enumerate(graphs):
            g = convert_node_labels_to_integers(g)
            s = "t"+str(i)+" "
            at_least_one_edge = False
            for v in g.nodes:
                for w in g.neighbors(v):
                    if int(str(v)) < int(str(w)):
                        s += id_to_str(int(str(v)))+"-"+id_to_str(int(str(w)))+";"
                        at_least_one_edge = True

            s = s[:-1] #remove last ";". but do we need this???
            s += "\n"
            if at_least_one_edge:
                file.write(s)
    return

def networkxToDISCDataGraphBatch(graphs, dir):

    for i, g in enumerate(graphs):

        g = convert_node_labels_to_integers(g, first_label=1) #DISC seems to expect 1,..,n labels and not 0,..,n-1

        with open(dir + "/" + "graph"+str(i)+".txt", 'w') as file:
            for v in g.nodes:
                for w in g.neighbors(v):
                    if int(str(v)) < int(str(w)):
                        file.write(str(v)+" "+str(w)+"\n")

if __name__ == '__main__':
    networkxToDISCPatternBatch([nx.path_graph(i) for i in range(10)], "temp.txt")
    networkxToDISCDataGraphBatch([nx.path_graph(i) for i in range(10)], "temp")






import graphgenerators
import homlib
import numpy as np
from homlib import Graph, hom
import GraphDataToGraphList as gi
import networkx as nx
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import grakel

def erGraph(n, p=0.5):
    G = Graph(n)
    for i in range(n):
        for j in range(i+1,n):
            if np.random.rand() <= p:
                G.addEdge(i,j)
    return G

def nx2Graph(g: nx.Graph):
    G = Graph(g.number_of_nodes())
    for e in g.edges:
        G.addEdge(e[0], e[1])
    return G

def kernel_alignment(g1, g2):
    ''' see, e.g. Equation (1) in 
    Tinghua Wang, Dongyan Zhao, Shengfeng Tian: 
    An overview of kernel alignment and its applications.
    Artif Intell Rev (2015) 43:179–192
    DOI 10.1007/s10462-012-9369-4
    https://link.springer.com/content/pdf/10.1007%2Fs10462-012-9369-4.pdf '''
    g1 = g1.flatten()
    g2 = g2.flatten()
    
    f11 = np.dot(g1, g1)
    f22 = np.dot(g2, g2)
    f12 = np.dot(g1, g2)
    
    return f12 / np.sqrt(f11 * f22)


def embedG(G, patterns):
    return np.array([hom(P, G) for P in patterns])


# G = erGraph(20, p=0.5)
patterns = [erGraph(n) for n in np.random.randint(1, 5, 100)]


path = '/home/pascal/Documents/DS_all/'
db = 'MUTAG'
nxgraphs, labels, _ = gi.graph_data_to_graph_list(path, db)

# print(labels)

graphs = [nx2Graph(g) for g in nxgraphs]

embeddings = np.vstack([embedG(G, patterns).reshape([1, -1]) for G in graphs])

# print(embeddings)

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.33, random_state=42)

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
print('accuracy of dt on test:', accuracy_score(y_pred, y_test))

WLKernel = grakel.WeisfeilerLehman()
gram_WL = WLKernel.fit_transform([grakel.Graph(nx.adjacency_matrix(g), {i:1 for i in range(g.number_of_nodes())}) for g in nxgraphs])
gram_hom = embeddings @ embeddings.T
gram_perfect = np.array(labels).reshape([-1,1]) @ np.array(labels).reshape([1,-1])

print('alignment wl-hom', kernel_alignment(gram_WL, gram_hom))
print('alignment wl-opt', kernel_alignment(gram_WL, gram_perfect))
print('alignment hom-opt', kernel_alignment(gram_perfect, gram_hom))
print('alignment hom-2hom', kernel_alignment(gram_perfect, 2* gram_perfect))


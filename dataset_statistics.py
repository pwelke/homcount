import numpy as np
from ghc.utils.data import load_data_for_json

graphs, _, y, metas = load_data_for_json("PAULUS25", 'data/graphdbs')

classes, counts = np.unique(y, return_counts=True)
print(f'classes = {classes}\ncounts = {counts}')


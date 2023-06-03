from zipfile import ZipFile
from os import makedirs
from os.path import join
import json
from ghc.utils.data import load_data

def convert_from_paulus(save_path:str='data/graphdbs/', source_path='dataset_conversion/PAULUS25.zip'):
    makedirs(save_path, exist_ok=True)
    with ZipFile(source_path) as archive:
        archive.extractall(path=save_path)


def make_meta(save_path:str='data/graphdbs/'):
    makedirs(save_path, exist_ok=True)
    metas = list()
    global_idx = 0

    for split in ['full']:

        graphs, X, y = load_data('PAULUS25', save_path)

        for i, g in enumerate(graphs):
            
            meta = {'vertices': len(g.nodes), 
                    'edges': len(g.edges),
                    'split': split,
                    'idx_in_split': i, 
                    'idx': global_idx}
            metas.append(meta)

            global_idx += 1
    
    with open(join(save_path, 'PAULUS25' + '.meta'), 'w') as f:
        json.dump(metas, f)   


if __name__ == '__main__':
    convert_from_paulus()
    make_meta()
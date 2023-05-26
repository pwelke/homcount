from zipfile import ZipFile
from os import makedirs

def convert_from_csl(save_path:str='data/graphdbs/', source_path='dataset_conversion/CSL.zip'):
    makedirs(save_path, exist_ok=True)
    with ZipFile(source_path) as archive:
        archive.extractall(path=save_path)

if __name__ == '__main__':
    convert_from_csl()
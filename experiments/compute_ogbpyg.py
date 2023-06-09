import subprocess
import itertools
import sys
from os.path import join
import hashlib 

# parameters to iterate over
cwd = './'

executables = ['pattern_extractors/hom.py', ] 

datasets = ['ogbg-moltox21',
            'ogbg-molesol',
            'ogbg-molbace',
            'ogbg-molclintox',
            'ogbg-molbbbp',
            'ogbg-molsider',
            'ogbg-moltoxcast',
            'ogbg-mollipo',
            'ogbg-molhiv',
            'ZINC_subset']

run_ids = ['run1', 'run2','run3', 'run4', 'run5', 'run6', 'run7', 'run8', 'run9', 'run10']

pattern_counts = [50] 

hom_types = ['full_kernel'] # choices: min_kernel, full_kernel

hom_size = 'max'
dloc = 'data'


# download and preprocess all datasets
args = ['python', join('dataset_conversion', 'import_ogbg.py')]
subprocess.run(args, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr, check=True)
args = ['python', join('dataset_conversion', 'import_TUDatasets.py')]
subprocess.run(args, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr, check=True)

# a deterministic hash function returning a 32 bit integer value for a given utf-8 string
hashfct = lambda x: str(int(hashlib.sha1(bytes(x, 'utf-8')).hexdigest(), 16) & 0xFFFFFFFF)

for run_id, dataset, executable, pattern_count, hom_type in itertools.product(run_ids, datasets, executables, pattern_counts, hom_types):
    print(f'{run_id}: {dataset} {executable}')    
    args = ['python', executable, 
            '--data', dataset,
            '--seed', hashfct(run_id),
            '--dloc', join(dloc, 'graphdbs'),
            '--oloc', join(dloc, 'homcount'),
            '--pattern_count', str(pattern_count),
            '--run_id', run_id,
            '--hom_type', hom_type,
            '--hom_size', '-1', # -1: select largest pattern size to be equal to largest graph in training set
            ]
    print(args)
    subprocess.run(args, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr, check=True)

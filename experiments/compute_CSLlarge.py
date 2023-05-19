import subprocess
import itertools
import sys
import os
import hashlib
from ghc.utils.converter import file_overflow_filter, file_singleton_filter

# parameters to iterate over
cwd = './'
dloc = 'graph-homomorphism-network/data/'

datasets = ['CSL']

executables = ['models/svm.py', 'models/mlp.py'] 

run_ids = ['run1', 'run2','run3', 'run4', 'run5', 'run6', 'run7', 'run8', 'run9', 'run10']

pattern_counts = [20, 50, 100, 150, 200]

hom_types = ['min_kernel'] # is actually min_kernel, but right now on existing data

# a deterministic hash function returning a 32 bit integer value for a given utf-8 string
hashfct = lambda x: str(int(hashlib.sha1(bytes(x, 'utf-8')).hexdigest(), 16) & 0xFFFFFFFF)


for run_id, dataset, executable, pattern_count, hom_type in itertools.product(run_ids, datasets, executables, pattern_counts, hom_types):
    print(f'{run_id}: {dataset} {executable}')
    args = ['python', executable, 
            '--data', dataset,
            '--seed', hashfct(run_id),
            '--dloc', dloc,
            '--pattern_count', str(pattern_count),
            '--run_id', run_id,
            '--hom_type', hom_type,
            '--hom_size', '-1',
            '--grid_search',
            ]
    subprocess.run(args, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr, check=True)

file_overflow_filter(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc + 'precompute')
file_singleton_filter(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc + 'precompute')
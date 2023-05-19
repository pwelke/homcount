import subprocess
import itertools
import sys
import os
import hashlib
from ghc.utils.converter import file_overflow_filter, file_singleton_filter

# parameters to iterate over
cwd = './'
dloc = 'graph-homomorphism-network/data/'


datasets = [
            'ogbg-moltox21',
            'ogbg-molesol',
            'ogbg-molbace',
            'ogbg-molclintox',
            'ogbg-molbbbp',
            'ogbg-molsider',
            'ogbg-moltoxcast',
            'ogbg-mollipo',
            'ZINC_subset',
            'ogbg-molhiv',
            ]

executables = ['models/fabian_extraction.py', ] 

run_ids = ['run1']

pattern_counts = [1,2,3,4,5] 

hom_types = ['wl_kernel']

hom_size = 'max'


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
            ]
    subprocess.run(args, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr, check=True)

file_overflow_filter(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc + 'precompute')
file_singleton_filter(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc + 'precompute')
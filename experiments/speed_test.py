import subprocess
import itertools
import sys
import os
import hashlib

# parameters to iterate over
cwd = './'

datasets = ['MUTAG'] #, 'CSL', 'PAULUS25', 'BZR', 'IMDBBINARY', 'IMDBMULTI', 'REDDIT-BINARY', 'NCI1', 'ENZYMES', 'DD', 'COLLAB']

executables = ['graph-homomorphism-network/models/embedding.py']

run_ids = ['run1','run2','run3', 'run4', 'run5', 'run6', 'run7', 'run8', 'run9', 'run10']

pattern_counts = [30, ] #10, 50, 100, 200]

hom_types = {'tree+cycle': '6', 'random_ktree': '-1',}

# a deterministic hash function returning a 32 bit integer value for a given utf-8 string
hashfct = lambda x: str(int(hashlib.sha1(bytes(x, 'utf-8')).hexdigest(), 16) & 0xFFFFFFFF)


for run_id, dataset, executable, pattern_count, hom_type in itertools.product(run_ids, datasets, executables, pattern_counts, hom_types):
    print(f'{run_id}: {dataset} {executable}')
    args = ['python', executable, 
            '--data', dataset,
            '--seed', hashfct(run_id),
            '--dloc', 'graph-homomorphism-network/data',
            '--pattern_count', str(pattern_count),
            '--run_id', run_id,
            '--hom_type', hom_type,
            '--hom_size', hom_types[hom_type],
            ]
    subprocess.run(args, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr, check=True)

import sys
import json
import itertools
from ghc.utils.data import load_json, save_json
import numpy as np


def filter_overflow(patterns, sizes):
    minval = np.min(patterns, axis=0)
    patterns = patterns[:, minval >= 0]
    if isinstance(sizes, list):
        sizes = [s for s,m in zip(sizes, minval) if m >= 0]
    else:
        sizes = sizes[minval >= 0]

    if patterns.shape[1] > 0:
        return patterns, sizes
    else: 
        # if nothing worked, return zeros
        return np.zeros([patterns.shape[0], 1]), np.zeros(1)

def filter_singletons(patterns, sizes):
    larger = (sizes > 2)
    # first column contains the singleton counts that we want to keep
    larger[0] = True
    # second column contains the single edge counts that we want to keep
    larger[1] = True
    patterns = patterns[:, larger]
    sizes = sizes[larger]

    if patterns.shape[1] > 0:
        return patterns, sizes
    else: 
        # if nothing worked, return zeros
        return np.zeros([patterns.shape[0], 1]), np.zeros(1)


def file_overflow_filter(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc):
    for run_id, dataset, pattern_count, hom_type in itertools.product(run_ids, datasets, pattern_counts, hom_types):

        try:
            meta = load_json(dataset.upper(), hom_type, hom_size, pattern_count, run_id, dloc)

            pattern_sizes = np.array(meta['pattern_sizes'])
            data = meta['data']
            features = np.array([x['counts'] for x in data], dtype=float)

            size_before = pattern_sizes.shape[0]

            features, pattern_sizes = filter_overflow(features, pattern_sizes)

            size_after = pattern_sizes.shape[0]

            meta['pattern_sizes'] = pattern_sizes.tolist()
            for x, f in zip(data, features):
                x['counts'] = f.tolist()

            print(f'{dataset} {size_before}->{size_after}, min={np.min(features)}')

            save_json(meta, dataset.upper(), hom_type, hom_size, pattern_count, run_id, dloc, suffix='overflow_filtered')
        except FileNotFoundError:
            pass


def file_singleton_filter(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc):
    for run_id, dataset, pattern_count, hom_type in itertools.product(run_ids, datasets, pattern_counts, hom_types):

        try:
            meta = load_json(dataset.upper(), hom_type, hom_size, pattern_count, run_id, dloc, suffix='overflow_filtered')

            pattern_sizes = np.array(meta['pattern_sizes'])
            data = meta['data']
            features = np.array([x['counts'] for x in data], dtype=float)

            size_before = pattern_sizes.shape[0]

            features, pattern_sizes = filter_singletons(features, pattern_sizes)

            size_after = pattern_sizes.shape[0]

            meta['pattern_sizes'] = pattern_sizes.tolist()
            for x, f in zip(data, features):
                x['counts'] = f.tolist()

            print(f'{dataset} {size_before}->{size_after}, min={np.min(features)}')

            save_json(meta, dataset.upper(), hom_type, hom_size, pattern_count, run_id, dloc, suffix='singleton_filtered')
        except FileNotFoundError:
            pass

def file_homdensity_filter(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc):

    for run_id, dataset, pattern_count, hom_type in itertools.product(run_ids, datasets, pattern_counts, hom_types):

        try:

            meta = load_json(dataset.upper(), hom_type, hom_size, pattern_count, run_id, dloc)

            pattern_sizes = meta['pattern_sizes']
            data = meta['data']

            def density(n, counts, sizes):
                return [c / (n ** s) for c,s in zip(counts, sizes)]

            for x in data:
                hom_density = density(x['vertices'], x['counts'], pattern_sizes)
                x['counts'] = hom_density

            save_json(meta, dataset.upper(), hom_type, hom_size, pattern_count, run_id, dloc, suffix='densities')
        except FileNotFoundError:
            pass # we don't process whats not there
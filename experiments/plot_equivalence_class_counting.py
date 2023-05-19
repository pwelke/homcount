import sys
sys.path.append('graph-homomorphism-network/src')
from ghc.utils.data import load_json, save_json
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

def file_equivalence_plot(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc):

    results = pd.DataFrame(columns=['run_id', 
                            'dataset',
                            'pattern_count',
                            'hom_type', 
                            'equivalence_classes', 
                            'n_patterns',
                            'n_examples'])

    for run_id, dataset, pattern_count, hom_type in itertools.product(run_ids, datasets, pattern_counts, hom_types):

        try:
            meta = load_json(dataset.upper(), hom_type, hom_size, pattern_count, run_id, dloc, suffix='singleton_filtered')

            pattern_sizes = np.array(meta['pattern_sizes'])
            data = meta['data']
            features = np.array([x['counts'] for x in data], dtype=float)

            equivalence_classes = [np.unique(features[:,:i], axis=0).shape[0] for i in range(len(pattern_sizes))]
            print(len(equivalence_classes))
            # results.append({'run_id': run_id, 
            #                 'dataset': dataset,
            #                 'pattern_count': pattern_count,
            #                 'hom_type': hom_type, 
            #                 'equivalence_classes': tuple(np.array(equivalence_classes)), 
            #                 'n_patterns': pattern_sizes.shape[0],
            #                 'n_examples': features.shape[0]})
            results = pd.concat([results, pd.DataFrame([[run_id, dataset, pattern_count, hom_type, [np.array(equivalence_classes)], pattern_sizes.shape[0], features.shape[0]]], columns=results.columns)], ignore_index=True)

        except FileNotFoundError:
            pass

    return results


def mean_different_length_array(list_of_arrays, new_len=54, pad_value=-1):
    print(f'i am getting {list_of_arrays}')
    # create array of shape [len(list_of_arrays), new_len), filling up shorter arrays with pad_value
    pad_cat = np.stack([np.pad(obj[0], pad_width=(0,new_len-obj[0].shape[0]), mode='constant', constant_values=pad_value) for obj in list_of_arrays])
    # compute mean for each column, ignoring cells containing pad_value
    mean = np.mean(pad_cat, axis=0, where=pad_cat != pad_value)
    return mean

def std_different_length_array(list_of_arrays, new_len=54, pad_value=-1):
    print(f'i am getting {list_of_arrays}')
    # create array of shape [len(list_of_arrays), new_len), filling up shorter arrays with pad_value
    pad_cat = np.stack([np.pad(obj[0], pad_width=(0,new_len-obj[0].shape[0]), mode='constant', constant_values=pad_value) for obj in list_of_arrays])
    # compute std for each column, ignoring cells containing pad_value
    std = np.std(pad_cat, axis=0, where=pad_cat != pad_value)
    return std


if __name__ == '__main__':

    # choose input data
    run_ids = ['run' + str(i+1) for i in range(9)]
    pattern_counts = [50] 
    hom_types = ['full_kernel']
    hom_size = 'max'
    dloc = '2023-03-07_ogbpyg_repetition/repeated_runs/'

    datasets = ['ogbg-moltox21',
                'ogbg-molesol',
                'ogbg-molbace',
                'ogbg-molclintox',
                'ogbg-molbbbp',
                'ogbg-molsider',
                'ogbg-moltoxcast',
                'ogbg-mollipo',
                'ogbg-molhiv',
                'ZINC_subset'
                ]


    # compute number of equivalence classes after each pattern for each dataset
    df = file_equivalence_plot(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc)

    # aggregation magic
    agg = df.groupby(['dataset','pattern_count','hom_type', 'n_examples'], sort=True).agg(mean=('equivalence_classes', mean_different_length_array), std=('equivalence_classes', std_different_length_array))

    # plotting

    colormapping = {
        'ogbg-moltox21':    'tab:blue',
        'ogbg-molesol':    'tab:orange',
        'ogbg-molbace':    'tab:green',
        'ogbg-molclintox':    'tab:red',
        'ogbg-molbbbp':    'tab:purple',
        'ogbg-molsider':    'tab:brown',
        'ogbg-moltoxcast':    'tab:pink',
        'ogbg-mollipo':    'tab:gray',
        'ogbg-molhiv':    'tab:olive',
        'ZINC_subset':    'tab:cyan'}


    plt.figure(figsize=[6,6], dpi=300)
    plt.set_cmap('tab10')
    for idx,tpl in agg.iterrows():
        plt.plot(np.arange(54) + 1, tpl['mean'] / idx[3], label=idx[0], color=colormapping[idx[0]])
    plt.xlabel('Number of Patterns')
    plt.ylabel('Equivalence Classes / Number of Examples')
    plt.legend()
    plt.xlim([0,50])
    plt.ylim([0,1])
    plt.savefig(dloc + 'equivalence_class_counting.pdf')
    plt.savefig(dloc + 'equivalence_class_counting.png')

    plt.show()






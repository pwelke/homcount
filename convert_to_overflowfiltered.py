from ghc.utils.converter import file_overflow_filter, file_singleton_filter

if __name__ == '__main__':

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
                'ZINC_subset']

    file_overflow_filter(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc)
    file_singleton_filter(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc)
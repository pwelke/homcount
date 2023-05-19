from ghc.utils.converter import file_homdensity_filter


if __name__ == "__main__":
    datasets = ['ogbg-moltox21',
            'ogbg-molesol',
            'ogbg-molbace',
            'ogbg-molclintox',
            'ogbg-molbbbp',
            'ogbg-molsider',
            'ogbg-moltoxcast',
            'ogbg-mollipo',
            'ogbg-molhiv',]

    run_ids = ['il3',]# 'il4',] #'run3', 'run4', 'run5', 'run6', 'run7', 'run8', 'run9', 'run10']

    pattern_counts = [50,] #[30, ] #10, 50, 100, 200]

    hom_types = ['full_kernel']

    hom_size = 'max'

    dloc = 'forFabian/2023-01-12_fixedreps/'

    file_homdensity_filter(run_ids, datasets, pattern_counts, hom_types, hom_size, dloc)

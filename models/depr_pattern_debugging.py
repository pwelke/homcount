import os
import uuid
import argparse
import pickle as pkl
from time import time
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('graph-homomorphism-network/src')

from ghc.homomorphism import get_hom_profile
from ghc.utils.data import load_data, load_precompute, save_precompute,\
                           load_folds, augment_data, precompute_patterns_file_handle,\
                           load_data_for_json, hom2json, save_json, load_precompute_patterns
from ghc.utils.ml import accuracy
import sys


if __name__ == "__main__":

    hom_types = get_hom_profile(None)

    #### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern_count', type=int, default=50)
    parser.add_argument('--run_id', type=str, default=0)
    parser.add_argument('--hom_size', type=int, default=6)

    parser.add_argument('--data', default='MUTAG')
    parser.add_argument('--hom_type', type=str, choices=hom_types)
    parser.add_argument('--dloc', type=str, default="./data")
    parser.add_argument('--oloc', type=str, default="./data")

    # arguments for compatibility reasons which are ignored
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.00005)
    parser.add_argument('--hids', type=int, nargs='+', default=[64, 64, 64])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--cuda', action="store_true", default=False)
    parser.add_argument('--verbose', action="store_true", default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--log_period", type=int, default=200)



    args = parser.parse_args()

    if args.hom_size == -1:
        args.hom_size = 'max' # use maximum graph size in database
        
    
    #### Setup checkpoints and precompute
    os.makedirs("./checkpoints/", exist_ok=True)
    os.makedirs(args.oloc, exist_ok=True)
    
    #### Load data and compute homomorphism
    graphs, _, y, metas = load_data_for_json(args.data.upper(), args.dloc)

    hom_func = get_hom_profile('large_pattern')


    with precompute_patterns_file_handle(args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id, args.oloc) as f:
        homX = hom_func(graphs, 
                        size=args.hom_size, 
                        density=False, 
                        seed=args.seed, 
                        pattern_count=args.pattern_count, 
                        pattern_file=f,
                        )
    save_precompute(homX, args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id, args.oloc)

    metas = hom2json(metas, homX, y)
    pattern_sizes = [len(p.nodes) for p in load_precompute_patterns(args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id, args.oloc)]
    metas = {'pattern_sizes': pattern_sizes, 'data': metas}
    save_json(metas, args.data.upper(), 'large_pattern', args.hom_size, args.pattern_count, args.run_id,
                    args.oloc)
    
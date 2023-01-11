import os
import uuid
import argparse
import pickle as pkl
from time import time
import numpy as np
from tqdm import tqdm
from ghc.homomorphism import get_hom_profile
from ghc.utils.data import load_data, load_precompute, save_precompute,\
                           load_folds, augment_data, precompute_patterns_file_handle
from ghc.utils.ml import accuracy
import sys
import json

def load_data_for_json(fname, dloc):
    graphs, feats, y = load_data(fname, dloc)
    name = os.path.abspath(os.path.join(dloc, fname + '.meta'))
    with open(name, 'r') as f:
        metas = json.load(f)
    return graphs, feats, y, metas

def hom2json(metas, homX, ys):
    for meta, hom, y in zip(metas, homX, ys):
        meta['counts'] = hom.tolist()
        meta['y'] = y
    return metas

def save_json(meta, dataset, hom_type, hom_size, pattern_count, run_id, dloc):
    dataf = os.path.abspath(dloc)
    tmp_str = f"{dataf}/{dataset}_{hom_type}_{hom_size}_{pattern_count}_{run_id}.homson"
    with open(tmp_str, 'w') as f:
        json.dump(meta, f)

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
    os.makedirs(os.path.join(args.dloc, "precompute"), exist_ok=True)
    
    #### Load data and compute homomorphism
    graphs, _, y, metas = load_data_for_json(args.data.upper(), args.dloc)
    hom_func = get_hom_profile(args.hom_type)
    try:
        homX = load_precompute(args.data.upper(),
                                args.hom_type,
                                args.hom_size,
                                args.pattern_count,
                                args.run_id,
                                os.path.join(args.dloc, "precompute"))
        print(f'({args.data.upper()},{args.hom_type},{args.hom_size},{args.pattern_count},{args.run_id},{os.path.join(args.dloc, "precompute")}) loads')
        

    except FileNotFoundError:
        # changed it to batch computation to not recompute the patterns each time
        with precompute_patterns_file_handle(args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id,
                        os.path.join(args.dloc, "precompute")) as f:
            homX = hom_func(graphs, 
                            size=args.hom_size, 
                            density=False, 
                            seed=args.seed, 
                            pattern_count=args.pattern_count, 
                            pattern_file=f,
                            )
        save_precompute(homX, args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id,
                        os.path.join(args.dloc, "precompute"))

        metas = hom2json(metas, homX, y)
        save_json(metas, args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id,
                            os.path.join(args.dloc, "precompute"))
    
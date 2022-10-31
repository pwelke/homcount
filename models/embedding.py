import argparse
import numpy as np
from tqdm import tqdm
from time import time
from ghc.utils.data import load_data, load_precompute, save_precompute,\
                           load_folds, create_folds, augment_data
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from ghc.homomorphism import get_hom_profile, random_tree_profile
from ghc.generate_k_tree import random_ktree_profile

from sklearn.preprocessing import StandardScaler
import os




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
    parser.add_argument('--seed', type=int, default=0)

    # arguments for compatibility reasons that are ignored
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
        
    
    #### Setup devices and random seeds
    device_id = "cpu"
    
    #### Setup checkpoints and precompute
    os.makedirs("./checkpoints/", exist_ok=True)
    os.makedirs(os.path.join(args.dloc, "precompute"), exist_ok=True)
   
    #### Load data and compute homomorphism
    # the middle parameter loads graph feature info and is ignored, for now
    graphs, _, y = load_data(args.data.upper(), args.dloc)
    y = y.flatten()
    try:
        splits = load_folds(args.data.upper(), args.dloc)
    except FileNotFoundError:
        splits = create_folds(args.data.upper(), args.dloc, y)

    hom_func = get_hom_profile(args.hom_type)

    # changed it to batch computation to not recompute the patterns each time
    tic = time()
    homX = hom_func(graphs, size=args.hom_size, density=False, seed=args.seed, pattern_count=args.pattern_count)
    toc = time()

    print(f"RUN {args.run_id} dims {homX.shape[0]} {homX.shape[1]} TIME {args.data.upper()} time {toc-tic:.4f} ")

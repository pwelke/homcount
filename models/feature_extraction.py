import os
import uuid
import argparse
import pickle as pkl
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from ghc.homomorphism import get_hom_profile, random_tree_profile, random_ktree_profile
from ghc.utils.data import load_data, load_precompute, save_precompute,\
                           load_folds, augment_data
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
    parser.add_argument('--drop_nodes', action="store_true", default=False)
    parser.add_argument('--drop_nodes_rate', type=int, default=1)
    parser.add_argument('--gen_per_graph', type=int, default=1)
    parser.add_argument('--dloc', type=str, default="./data")
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
        

    #### Setup devices and random seeds
    torch.manual_seed(args.seed)
    device_id = "cpu"
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device_id = "cuda:"+str(args.gpu_id)
    device = torch.device(device_id)
    
    #### Setup checkpoints and precompute
    os.makedirs("./checkpoints/", exist_ok=True)
    os.makedirs(os.path.join(args.dloc, "precompute"), exist_ok=True)
    
    #### Load data and compute homomorphism
    graphs, X, y = load_data(args.data.upper(), args.dloc)
    hom_func = get_hom_profile(args.hom_type)
    try:
        homX = load_precompute(args.data.upper(),
                        args.hom_type,
                        args.hom_size,
                        args.pattern_count,
                        args.run_id,
                        os.path.join(args.dloc, "precompute"))

    except FileNotFoundError:
        if X is not None:
            # changed it to batch computation to not recompute the patterns each time
            homX = hom_func(graphs, density=False, seed=args.seed, pattern_count=args.pattern_count, pattern_size=args.hom_size)
            save_precompute(homX, args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id,
                            os.path.join(args.dloc, "precompute"))

    
    
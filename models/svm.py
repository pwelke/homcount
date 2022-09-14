import argparse
import numpy as np
from tqdm import tqdm
from time import time
from ghc.utils.data import load_data, load_precompute, save_precompute,\
                           load_folds, augment_data
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
    # Hyperparams for SVM
    parser.add_argument("--C", type=float, help="SVC's C parameter.", default=1e4)
    parser.add_argument("--kernel", type=str, help="SVC kernel function.",
                        default="rbf")
    parser.add_argument("--degree", type=int, help="Degree of `poly` kernel.",
                        default=2)
    parser.add_argument("--gamma", type=float, help="SVC's gamma parameter.",
                        default=40.0)
    # Misc
    parser.add_argument("--num_run", type=int, default=3,
                        help="Number of experiments to run.")
    parser.add_argument("--grid_search", action="store_true", default=False)
    parser.add_argument("--gs_nfolds", type=int, default=5)
    parser.add_argument("--disable_hom", action="store_true", default=False)
    parser.add_argument("--f1avg", type=str, default="micro",
                        help="Average method for f1.")
    parser.add_argument("--scaler", type=str, default="standard",
                        help="Name of data scaler to use as the preprocessing step")

    parser.add_argument("--test_ratio", type=float, default=.1)

    Cs = np.logspace(-5, 6, 20)
    gammas = np.logspace(-5, 1, 5)
    class_weight = ['balanced']
    param_grid = {'C': Cs, 'gamma': gammas, 'class_weight': class_weight}

    args = parser.parse_args()


    if args.hom_size == -1:
        args.hom_size = 'max' # use maximum graph size in database
        
    
    #### Setup devices and random seeds
    device_id = "cpu"
    
    #### Setup checkpoints and precompute
    os.makedirs("./checkpoints/", exist_ok=True)
    os.makedirs(os.path.join(args.dloc, "precompute"), exist_ok=True)
   
    #### Load data and compute homomorphism
    graphs, X, y = load_data(args.data.upper(), args.dloc)
    y = y.flatten()
    try:
        splits = load_folds(args.data.upper(), args.dloc)
    except FileNotFoundError:
        splits = create_folds(args.data.upper(), args.dloc, X)

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
            homX = hom_func(graphs, size=args.hom_size, density=False, seed=args.seed, pattern_count=args.pattern_count)
            save_precompute(homX, args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id,
                            os.path.join(args.dloc, "precompute"))

    #### If data augmentation is enabled
    if args.verbose:
        print(homX[0])
    if args.drop_nodes:
        gen_graphs, gen_X, gen_y = augment_data(graphs, X, y,
                                                args.gen_per_graph,
                                                rate=args.drop_nodes_rate)
        gen_hom_X = [hom_func(g, size=args.hom_size,
                              density=False,
                              node_tags=gen_X) \
                     for g in tqdm(gen_graphs, desc="Hom (aug)")]
    X = np.array(homX)
    
    # Train SVC 
    print(X.shape)
    print("Training SVM...")
    svm_time = time()
    a_acc = []  # All accuracies of num_run
    for j in tqdm(range(args.num_run)):
        acc = []
        skf = StratifiedKFold(n_splits=int(1/args.test_ratio), shuffle=True)
        for train_idx, test_idx in skf.split(X, y): 
            X_train = X[train_idx]
            X_test = X[test_idx] 
            y_train = y[train_idx] 
            y_test = y[test_idx]
            # Fit a scaler to training data
            scaler = StandardScaler()
            scaler = scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            if args.grid_search:
                grid_search = GridSearchCV(SVC(kernel=args.kernel), param_grid, cv=args.gs_nfolds,
                                           n_jobs=8)
                grid_search.fit(X_train,y_train)
                print(grid_search.best_params_)
                clf = SVC(**grid_search.best_params_)
            else:
                clf = SVC(C=args.C, kernel=args.kernel, degree=args.degree, 
                          gamma=args.gamma, decision_function_shape='ovr',
                          random_state=None, class_weight='balanced')
            clf.fit(X_train, y_train)
            print("train", f1_score(y_pred=clf.predict(X_train), y_true=y_train))
            acc.append(f1_score(y_pred=clf.predict(X_test), 
                                y_true=y_test, average=args.f1avg))
            print("val", f1_score(y_pred=clf.predict(X_test),
                                y_true=y_test, average=args.f1avg))
        a_acc.extend(acc)
    svm_time = time() - svm_time
    print("Accuracy: {:.4f} +/- {:.4f}".format(np.mean(a_acc), np.std(a_acc)))

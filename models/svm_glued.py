import argparse
import numpy as np
from tqdm import tqdm
from time import time
from ghc.utils.data import load_data, load_precompute, save_precompute, load_folds, create_folds, augment_data
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from ghc.homomorphism import get_hom_profile, random_tree_profile
from ghc.generate_k_tree import random_ktree_profile
from ghc.utils.fast_weisfeiler_lehman import homsub_format_wl_nodelabels

from sklearn.preprocessing import StandardScaler
import os


def check_onehot(vertex_labels):
    for g in vertex_labels:
        uni = np.unique(g).__str__()
        if uni != '[0. 1.]':
            if uni != '[1]':
                print(f'ERROR: vertex features are not one-hot encoded: {uni}')


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
    
    # Hyperparams for SVM
    # makes all pams below obsolete
    parser.add_argument("--grid_search", action="store_true", default=False)

    # 
    parser.add_argument("--C", type=float, help="SVC's C parameter.", default=1e4)
    parser.add_argument("--kernel", type=str, help="SVC kernel function.",
                        default="rbf")
    parser.add_argument("--degree", type=int, help="Degree of `poly` kernel.",
                        default=2)
    parser.add_argument("--gamma", type=float, help="SVC's gamma parameter.",
                        default=40.0)
    parser.add_argument("--gs_nfolds", type=int, default=5)
    #parser.add_argument("--disable_hom", action="store_true", default=False)
    parser.add_argument("--f1avg", type=str, default="micro",
                        help="Average method for f1.")
    parser.add_argument("--scaler", type=str, default="standard",
                        help="Name of data scaler to use as the preprocessing step")

    Cs = np.logspace(start=-5, stop=6, num=20).tolist()
    gammas = np.logspace(start=-5, stop=1, num=7).tolist() + ['scale']
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
    # the middle parameter loads graph feature info and is ignored, for now
    graphs, vertex_labels, y = load_data(args.data.upper(), args.dloc)
    y = y.flatten()
    try:
        splits = load_folds(args.data.upper(), args.dloc)
    except FileNotFoundError:
        splits = create_folds(args.data.upper(), args.dloc, y)

    hom_func = get_hom_profile(args.hom_type)
    try:
        homX = load_precompute(args.data.upper(),
                        args.hom_type,
                        args.hom_size,
                        args.pattern_count,
                        args.run_id,
                        os.path.join(args.dloc, "precompute"))

    except FileNotFoundError:
        # changed it to batch computation to not recompute the patterns each time
        homX = hom_func(graphs, size=args.hom_size, density=False, seed=args.seed, pattern_count=args.pattern_count)
        save_precompute(homX, args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id,
                        os.path.join(args.dloc, "precompute"))


    # assumes vertex labels to be one-hot encoded
    check_onehot(vertex_labels)
    vertex_label_counts = np.array([np.sum(g, axis=0) for g in vertex_labels])
    if vertex_label_counts.ndim == 1:
        vertex_label_counts = vertex_label_counts.reshape([-1,1])

    wl_labels = homsub_format_wl_nodelabels(graphs, vertex_labels, n_iter=3)
    wl_label_counts = np.array([np.sum(g, axis=0) for g in wl_labels])
    
    X = np.hstack([np.array(homX), vertex_label_counts, wl_label_counts])
    
    # Train SVC 
    svm_time = time()
    a_acc = []  # All accuracies of num_run
    # for j in tqdm(range(args.num_run)):
    if True:
        acc = []
        # skf = StratifiedKFold(n_splits=int(1/args.test_ratio), shuffle=True)
        for train_idx, test_idx in tqdm(splits): # skf.split(X, y): 
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
                if args.verbose:
                    print(grid_search.best_params_)
                clf = SVC(**grid_search.best_params_)
            else:
                clf = SVC(C=args.C, kernel=args.kernel, degree=args.degree, 
                          gamma=args.gamma, decision_function_shape='ovr',
                          random_state=None, class_weight='balanced')
            clf.fit(X_train, y_train)
            # print("train", f1_score(y_pred=clf.predict(X_train), y_true=y_train))
            acc.append(accuracy_score(y_pred=clf.predict(X_test), y_true=y_test))
            # print("val", f1_score(y_pred=clf.predict(X_test),
                                # y_true=y_test, average=args.f1avg))
        a_acc.extend(acc)
    svm_time = time() - svm_time
    print(f"RUN {args.run_id} dims {X.shape[0]} {X.shape[1]} SVMGLUED {args.data.upper()} mean {np.mean(a_acc):.4f} std {np.std(a_acc):.4f}")

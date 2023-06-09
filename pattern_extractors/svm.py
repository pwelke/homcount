import argparse
import numpy as np
from tqdm import tqdm
from time import time

from ghc.utils.data import load_data_for_json, load_precompute, save_precompute,\
                           load_folds, create_folds, precompute_patterns_file_handle,\
                           hom2json, save_json, load_precompute_patterns

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from ghc.homomorphism import get_hom_profile

from sklearn.preprocessing import StandardScaler
import os


def compute_svm(passed_args=None):
    hom_types = get_hom_profile(None)

    #### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern_count', type=int, default=50) # if -1 then adjust by expressive power
    parser.add_argument('--run_id', type=str, default=0)
    parser.add_argument('--hom_size', type=int, default=6)

    parser.add_argument('--data', default='MUTAG')
    parser.add_argument('--hom_type', type=str, choices=hom_types)
    parser.add_argument('--dloc', type=str, default="./data")
    parser.add_argument('--oloc', type=str, default="./data")
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

    # Load partial args instead of command line args (if they are given)
    if passed_args is not None:
        # Transform dict to list of args
        list_args = []
        for key,value in passed_args.items():
            # The case with "" happens if we want to pass an argument that has no parameter
            list_args += [key, str(value)]

        args = parser.parse_args(list_args)
    else:
        args = parser.parse_args()

    # Gridsearch ranges (will be used if grid-search arg is set)
    Cs = np.logspace(start=-5, stop=6, num=20).tolist()
    gammas = np.logspace(start=-5, stop=1, num=7).tolist() + ['scale']
    class_weight = ['balanced']
    param_grid = {'C': Cs, 'gamma': gammas, 'class_weight': class_weight}

    if args.hom_size == -1:
        args.hom_size = 'max' # use maximum graph size in database
    
    #### Setup checkpoints and precompute
    os.makedirs(args.oloc, exist_ok=True)

    #### Load data and compute homomorphism
    # the middle parameter loads graph feature info and is ignored, for now
    graphs, _, y, metas = load_data_for_json(args.data.upper(), args.dloc)
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
                        args.oloc)

    except FileNotFoundError:
        # changed it to batch computation to not recompute the patterns each time
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
        try:
            pattern_sizes = [len(p.nodes) for p in load_precompute_patterns(args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id, args.oloc)]
        except EOFError:
            ## TODO careful: this is hacky and supposed to work for for WL patterns, that don't have any size we want to compute
            pattern_sizes = [args.pattern_count for _ in range(homX.shape[1])]

        metas = {'pattern_sizes': pattern_sizes, 'data': metas}
        save_json(metas, args.data.upper(), args.hom_type, args.hom_size, args.pattern_count, args.run_id, args.oloc)

    X = np.array(homX)
    
    # Train SVC 
    svm_time = time()

    acc = []
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
        acc.append(accuracy_score(y_pred=clf.predict(X_test), y_true=y_test))
        # print("val", f1_score(y_pred=clf.predict(X_test), y_true=y_test, average=args.f1avg))

    svm_time = time() - svm_time
    print(f"RUN {args.run_id} dims {X.shape[0]} {X.shape[1]} SVM {args.data.upper()} mean {np.mean(acc):.4f} std {np.std(acc):.4f}")


if __name__ == "__main__":
    compute_svm(passed_args=None)

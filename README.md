# Expectation Complete Graph Representations

This repository contains the code to compute graph representations that are complete in expectation. 
If you use this code in your work or find it useful in any other way, please consider accepting our paper :).

# Installation

## System Setup
Ensure that you have python installed, cmake, and a recent c++ compiler available.
Currently, java is also a dependency, but it is superfluous.

## Clone Repo

To run the code in this repository, clone it somewhere and initialize the git submodules
```
git clone git@anonymous.4open.science/r/HomCount_Counts_ICML_2023.git
cd homcount
git submodule init
git submodule update
```

## Building HomSub

To compile c++ part, enter the `HomSub` and compile the code

```
cd HomSub
git submodule init
git submodule update
sh build-third-party.sh
sh build.sh
```

## Python Setup

Create a virtual environment, using python 3.7 (!) and install dependencies, e.g. with anaconda:

```
cd graph-homomorphism-network
conda create -n expectation_complete python==3.7
conda activate expectation_complete
pip install -r requirements_dev.txt
python setup.py install
```

# Compute Embeddings and Evaluate Results

- Download data from [here](https://drive.google.com/file/d/15w7UyqG_MjCqdRL2fA87m7-vanjddKNh/view?usp=sharing) and unzip it into `graph-homomorphism-network/data`.
- Run (in the virtual environment) `python experiments.py`, to only compute the embeddings of the selected datasets (if not already done) and save them in `graph-homomorphism-network/data/precompute`.
- Run (in the virtual environment) `python evaluation.py`, to compute a number of embeddings of the selected datasets (if not already done) and save them in `graph-homomorphism-network/data/precompute`. After that, run 10-fold cross validations for the MLP and SVM classifiers. 
- Note that there is currently a race condition with a temp file in . Hence, you cannot run multiple experiments simultaneously on the project folder. A workaroud would be to copy the full project folder multiple times.
- Currently, the average accuracies have to be manually collected from the output of evaluation.py.


# README of graph-homomorphism-network

TODO: This is the old readme of the merged project and should be integrated here


# Graph Homomorphism Sampling and Counting

Our code to sample patterns and then count homomorphisms started as a fork
of the [code for the Graph Homomorphism Convolution](https://github.com/gear/graph-homomorphism-network) paper. 

Below, you can find the original README of that repository, but be aware that we have changed quite a few things, in particular `mlp.py` and `svm.py`, as well as large parts of the ghc package.


## Graph Homomorphism Convolution
Proof of concept for Graph Homomorphism Convolution.
http://arxiv.org/abs/2005.01214 (ICML'20)

Note: Code for left homomorphism is for our ICML'20 paper.
Code for right homomorphism is our continued work.

### Run experiments
Experiment scripts are placed in the top level of this repository and named 
by the machine learning model. In general, a 10-fold CV score is reported.
For example,
```
python models/mlp.py --data mutag --hom_type tree --hom_size 6 
python models/mlp.py --data mutag --hom_type labeled_tree --hom_size 6
```
The 10-fold splits for MUTAG, COLLAB, IMDBBINARY, IMDBMULTI, NCI1, PROTEINS, and
PTC are taken from `weihua916/powerful-gnns`. The others are generated with 
`sklearn.model_selection.StratifiedKFold` at random seed 0.

Note: CPU run will cause some segmentation fault upon script exit.

Cite us as:
```
@InProceedings{
    nt20ghc, 
    title = {Graph Homomorphism Convolution}, 
    author = {NT, Hoang and Maehara, Takanori}, 
    booktitle = {Proceedings of the 37th International Conference on Machine Learning}, 
    pages = {7306--7316}, 
    year = {2020}, 
    editor = {Hal Daumé III and Aarti Singh}, 
    volume = {119}, 
    series = {Proceedings of Machine Learning Research}, 
    address = {Virtual}, 
    month = {13--18 Jul}, 
    publisher = {PMLR}, 
    pdf = {http://proceedings.mlr.press/v119/nguyen20c/nguyen20c.pdf},, 
    url = {http://proceedings.mlr.press/v119/nguyen20c.html}, 
}
```

Note: There is a bug in homlib for `atlas[100:]`.







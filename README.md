# Expectation Complete Graph Representations

This repository contains the code to compute graph representations that are complete in expectation. 
Its sister repository [HomCountGNNs](https://github.com/ocatias/HomCountGNNs) contains the code to train GNNs using the expectation-complete graph representations.
If you use this code in your work or find it useful in any other way, please consider citing our paper.

```
@inproceedings{welke2023expectation,
publicationtype = {preprint},
  pdf = {https://openreview.net/pdf?id=ppgRPC14uI},
  code = {https://github.com/pwelke/homcount},
  reviews = {https://openreview.net/forum?id=ppgRPC14uI},
  venuetype = {conference},
  venueurl = {https://icml.cc/2023},
  author       = {Pascal Welke* and Maximilian Thiessen* and Fabian Jogl and Thomas Gärtner},
  title        = {Expectation-Complete Graph Representations with Homomorphisms},
  booktitle    = {International Conference on Machine Learning ({ICML})},
  year = {2023},  
}
```

This repository started as a fork of [graph-homomorphism-convolution](https://github.com/gear/graph-homomorphism-network). Thanks to NT Hoang!


# Installation

## System Setup
Ensure that you have python installed, cmake, and a recent c++ compiler available.

## Clone Repo

To run the code in this repository, clone it somewhere and initialize all git submodules
```
git clone --recurse-submodules https://github.com/pwelke/homcount
cd homcount
```

## Building HomSub

To compile c++ part, enter the `HomSub` folder and compile the code

```
cd HomSub
sh build-third-party.sh
sh build.sh
cd ..
```

## Python Setup

Create a virtual environment, using python >=3.7 and install dependencies, e.g. with anaconda:
Make sure, you are in the base directory, again.

```
conda create -n expectation_complete
conda activate expectation_complete
pip install -r requirements.txt
```

The dependency ogb (Open Graph Benchmark) is only necessary if you want to download the ogb-provided datasets ogbg-mol*.

# Compute Embeddings and Evaluate Results

## Downloading Data
You can either download all data files used for the experiments in our paper via a single link, or run scripts that download them from different sources.

- [Download the graph datasets from here](https://drive.google.com/file/d/15w7UyqG_MjCqdRL2fA87m7-vanjddKNh/view?usp=sharing) and unzip them into `data/graphbds`.
- Alternatively, run (in the virtual environment) the scripts in `dataset_conversion`. These create the required datasets in the correct location. 
If you need to transform your own graphs into the required input format, have a look at the files in `dataset_conversion`. Dataset imports from the Open Graph Benchmark or from Pytorch Geometric should be possible more or less straight away. 

## Downloading Embeddings
As our embeddings are inherently randomized and as it is difficult to reliably reproduce randomized experiments on different hardware and software stacks, we also provide the embeddings we have used for our experiments. [You can download the embeddings here]()

## Rerunning our Experiments 
- Run (in the virtual environment) `python experiments/compute_ogbpyg.py`, to compute the embeddings of the selected datasets (if not already done) and save them in `data/homcount`.
    - `data/homcount` now contains files with the extension `.homson` that are json formatted and contain information on pattern sizes and, for each graph, the computed pattern counts. 
    - patterns are stored as pickled networkx graphs in files with extension `.patterns`
    - homomorphism counts are also stored in binary numpy format in files with extension `.hom`
- Run (in the virtual environment) `python experiments/compute_TUDatasets.py`, to compute a number of embeddings of the selected datasets (if not already done) and save them in `data/homcount`. After that, the script runs 10-fold cross validations for the MLP and SVM classifiers. 
- Note that there is currently a race condition with a temp file in the C++ part of the homomorphism computation. Hence, you cannot run multiple experiments simultaneously on the project folder. A workaroud would be to copy the full project folder multiple times.
- Currently, the average accuracies have to be manually collected from the output of `experiments/compute_TUDatasets.py`.
- GNN training and performance evaluation is delegated to the code in [HomCountGNNs](https://github.com/ocatias/HomCountGNNs)


## Computation of Homomorphism Counts from Python

The file `pattern_extractors/hom.py` contains a function `compute_hom`. You can call it with your parameters of choice to go from graph database to computed homomorphism patterns.
If you need to transform your graphs into the required input format, have a look at the files in `dataset_conversion`. Dataset imports from the Open Graph Benchmark or from Pytorch Geometric should be possible more or less straight away. 




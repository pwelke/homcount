
# Graph Homomorphism Sampling and Counting

Our code to sample patterns and then count homomorphisms started as a fork
of the [code for the Graph Homomorphism Convolution](https://github.com/gear/graph-homomorphism-network) paper. 

Below, you can find the original README of that repository, but be aware that we have changed quite a few things, in particular `mlp.py` and `svm.py`.


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

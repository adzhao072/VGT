# Voronoi Graph Traversing (VGT)
Voronoi Graph Traversing (VGT) is a high-dimensional Bayesian optimization method designed to handle the high-dimensional input spaces ranging from hundreds to one thousand dimensions. VGT employs a Voronoi diagram to partition the design space and transform it into an undirected Voronoi graph. VGT explores the search space through iterative path selection, promising cell sampling, and graph expansion operations.

Please cite this package as follows: 

```
@inproceedings{
VGT-uai2024,
title={Exploring High-dimensional Search Space via Voronoi Graph Traversing},
author={Zhao, Aidong and Zhao, Xuyang and Gu, Tianchen and Bi, Zhaori and Sun, Xinwei and Yan, Changhao and Yang, Fan and Zhou, Dian and Zeng, Xuan},
booktitle={The 40th Conference on Uncertainty in Artificial Intelligence},
year={2024},
url={https://openreview.net/forum?id=Phyo9GzgWd}
}
```


## Dependencies
--------------

    - Python == 3.9.16
    - numpy == 1.24.2
    - scipy == 1.10.1
    - setuptools == 53.0.0
    - torch == 2.0.0
    - gpytorch == 1.10




## Examples

import numpy as np
from benchmarks.synthetic_functions import Ackley


#input dimensions
dims = 100

#normalize the input search space
lb = np.zeros(dims)
ub = np.ones(dims)

#objective function
f = Ackley 

#random initial samples
n_init = 50

#maximal iterations
max_iter = 2050

#number of neighbors for KNN
N_neighbor = 60

#exploration-exploitation balance
Cp = 0.02

# Search with VGT 
num_samples=20000



from VGT import VGT

agent = VGT(f,lb,ub,n_init,max_iter,N_neighbor=N_neighbor,Cp= Cp,num_samples = num_samples)
agent.search()








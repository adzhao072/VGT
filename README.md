
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

# samples of acquisition function
num_samples=20000



from VGT import VGT

agent = VGT(f,lb,ub,n_init,max_iter,N_neighbor=N_neighbor,Cp= Cp,num_samples = num_samples)
agent.search()








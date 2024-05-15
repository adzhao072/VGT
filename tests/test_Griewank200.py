# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from benchmarks.synthetic_functions import Griewank


if os.path.exists('result.csv'):
    os.remove('result.csv')


dims = 100
lb = np.zeros(dims)
ub = np.ones(dims)

f = Griewank 

n_init = 50
max_iter = 2050
N_neighbor = 40
Cp = 0.01

from VGT import VGT


agent = VGT(f,lb,ub,n_init,max_iter,N_neighbor=N_neighbor,Cp= Cp)
agent.search()




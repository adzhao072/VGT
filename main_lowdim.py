# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from benchmarks.synthetic_functions import Hartmann6, Griewank,Ackley,RosenBrock,Rastrigin,Michalewicz,Levy


if os.path.exists('result.csv'):
    os.remove('result.csv')

dims = 5
lb = np.zeros(dims)
ub = np.ones(dims)

f = Ackley #Hartmann6 #RosenBrock # RosenBrock #Griewank #RosenBrock #Griewank #Levy #Levy #  RosenBrock # Griewank #RosenBrock #Michalewicz # #RosenBrock#Levy#Michalewicz#Hartmann6#Levy#Ackley#Michalewicz#Rastrigin#Ackley#Griewank#Rastrigin#Ackley#RosenBrock#Ackley

n_init = 10
max_iter = 250
#N_neighbor = 80
Cp = 150#0.2
num_samples=20000

use_approximation = True #False
N_neighbor = 20
from VGT import VGT


agent = VGT(f,lb,ub,n_init,max_iter,Cp= Cp,use_approximation = use_approximation,N_neighbor=N_neighbor, num_samples = num_samples)
agent.search()




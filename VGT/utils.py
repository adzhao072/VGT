




import numpy as np
from .LBFGS_torch import lbfgsb
import torch


def latin_hypercube(n, dims):
    points = np.zeros((n, dims))
    centers = np.arange(n)
    centers = centers / float(n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(np.arange(n))]

    perturbation = np.random.rand(n, dims) 
    perturbation = perturbation / float(n)
    points += perturbation
    return points

def to_size(x, lb, ub):
    return lb + (ub-lb) * x


def latin_hypercube_torch(n, dims, device):
    points = torch.zeros((n, dims),device = device)
    centers = torch.arange(0.0, n, device = device)
    centers = centers / float(n)
    for i in range(0, dims):
        points[:, i] = centers[torch.randperm(n,device = device)]

    perturbation = torch.rand(n, dims, device = device) 
    perturbation = perturbation / float(n)
    points += perturbation
    return points

def tosize_torch(x_tensor,bounds):
    return bounds[0] + (bounds[1]-bounds[0]) * x_tensor

def acq_max_msp(f, gradf, bounds, n_warmup=10000, n_iter=10):
    
    #t0 = time.time()
    device = bounds.device
    dims = bounds.size(1)
    rand_points = latin_hypercube_torch(n_warmup, dims, device)
    x_tries = tosize_torch(rand_points,bounds)
    ys = f(x_tries)
    idx = torch.argsort(ys)
    max_acq = ys[idx[0]]
    x_max = x_tries[idx[0]]
    #t1 = time.time()
    
    xseeds = x_tries[idx[:n_iter],:]

    for k in range(n_iter):
        optimizer = lbfgsb(lambda x:-f(x), lambda x: -gradf(x), bounds[0],bounds[1], device=device)
        xopt, yopt = optimizer.optimize(xseeds[k])
        
        if -yopt >= max_acq:
            x_max = xopt
            max_acq = -yopt
    #t2 = time.time()
    
    #print('warmup time = ',t1-t0,'bfgs time = ',t2-t1)
    return x_max

def acq_min_msp(f, gradf, bounds, n_warmup=10000, n_iter=5):
    device = bounds.device
    dims = bounds.size(1)
    rand_points = latin_hypercube(n_warmup, dims, device)
    x_tries = bounds[0] + (bounds[1]-bounds[0]) * rand_points
    ys = f(x_tries)
    idx = torch.argsort(-ys)
    min_acq = ys[idx[0]]
    x_min = x_tries[idx[0]]
    
    xseeds = x_tries[idx[:n_iter],:]
    for k in range(n_iter):
        optimizer = lbfgsb(f, gradf, bounds[0],bounds[1], device=device)
        xopt, yopt = optimizer.optimize(xseeds[k])
        if min_acq is None or yopt <= min_acq:
            x_min = xopt
            min_acq = yopt
    return x_min

   




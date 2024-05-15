# -*- coding: utf-8 -*-


import math

import gpytorch
import torch
import numpy as np
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

def norm_x(x,bounds):
    normed_x = (torch.atleast_2d(x)-bounds[0])/(bounds[1]-bounds[0])
    return normed_x

def tosize_x(x_norm,bounds):
    x = x_norm*(bounds[1]-bounds[0]) + bounds[0]
    return x

# GP Model
class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims, mean_y, std_y, bounds):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=2.5)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)
        self.likelihood = likelihood 
        self.mean_y = mean_y
        self.std_y = std_y
        self.min_y = train_y.min()
        self.bounds = bounds

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def predict(self, xtensor):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_tensor = norm_x(xtensor,self.bounds)
            predict = self.likelihood(self.__call__(x_tensor))
            mu = predict.mean * self.std_y + self.mean_y  
            std = torch.sqrt(predict.variance) * self.std_y 
        return mu,std
    
    def predict_mean(self, xtensor):
        x_tensor = norm_x(xtensor,self.bounds)
        predict = self.likelihood(self.__call__(x_tensor))
        mu = predict.mean
        return mu
    
    def predict_meanstd(self, xtensor):
        x_tensor = norm_x(xtensor,self.bounds)
        predict = self.likelihood(self.__call__(x_tensor))
        mu = predict.mean
        std = torch.sqrt(predict.variance)
        return mu,std
    
    def predict_grad(self, xtensor):
        return torch.autograd.functional.jacobian(self.predict_mean,xtensor).sum(dim = 0)
    
    def LCB(self, xtensor, kappa = 1.5):
        # min LCB acq
        with gpytorch.settings.fast_pred_var():
            x_tensor = norm_x(xtensor,self.bounds)
            predict = self.likelihood(self.__call__(x_tensor))
            mu = predict.mean  
            std = torch.sqrt(predict.variance)
            lcb = mu - kappa * std
        return lcb
    
    def LCB_nograd(self, xtensor, kappa = 1.5):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_tensor = norm_x(xtensor,self.bounds)
            predict = self.likelihood(self.__call__(x_tensor))
            mu = predict.mean 
            std = torch.sqrt(predict.variance)
            lcb = mu - kappa * std
        return lcb
    
    def grad_LCB(self, x_tensor, kappa = 1.5):
        return torch.autograd.functional.jacobian(self.LCB, x_tensor).sum(dim = 0)
        
    def EI(self, xtensor):
        x_tensor = norm_x(xtensor,self.bounds)
        predict = self.likelihood(self.__call__(x_tensor))
        mu = predict.mean  
        std = torch.sqrt(predict.variance)
        a = self.min_y - mu
        z = a/std
        return a * appro_normcdf(z) + std * normpdf(z)
    
    def EI_nograd(self, xtensor):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_tensor = norm_x(xtensor,self.bounds)
            predict = self.likelihood(self.__call__(x_tensor))
            mu = predict.mean 
            std = torch.sqrt(predict.variance)
            a = self.min_y - mu
            z = a/std
        return torch.maximum(a * appro_normcdf(z) + std * normpdf(z), torch.tensor(1e-17,device=a.device))
    
    def grad_EI(self, x_tensor):
        return torch.autograd.functional.jacobian(self.EI, x_tensor).sum(dim = 0)
    
    def PI(self, xtensor):
        x_tensor = norm_x(xtensor,self.bounds)
        predict = self.likelihood(self.__call__(x_tensor))
        mu = predict.mean  
        std = torch.sqrt(predict.variance)
        z = (self.min_y - mu)/std
        return appro_normcdf(z)
    
    def PI_nograd(self, xtensor):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_tensor = norm_x(xtensor,self.bounds)
            predict = self.likelihood(self.__call__(x_tensor))
            mu = predict.mean  
            std = torch.sqrt(predict.variance)
            z = (self.min_y - mu)/std
            
            #z = self.min_y - mu
            z[std >1] = -10
            #print(std)
            #z[std >3] = -1
            #print(z)
        return appro_normcdf(z)
    
    def grad_PI(self, x_tensor):
        return torch.autograd.functional.jacobian(self.PI, x_tensor).sum(dim = 0)

def appro_normcdf(x_tensor):
    #use Logistic function compute cdf sigma(1.702*x),
    #sigma(x) = 1/(1+exp(-x))
    return 1./(1+torch.exp(-1.702*x_tensor))

def normpdf(x_tensor):
    #1/sqrt(2*pi) = 0.39695
    return 0.39695*torch.exp(-0.5*x_tensor**2)


import time

def train_gp(train_x, train_y, use_ard, num_steps, bounds=None, hypers={}):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]
    
    t0 = time.time()
    mean_y = torch.mean(train_y)
    std_y, mean_y = torch.std_mean(train_y)
    normed_train_y = (train_y-mean_y) / std_y

    if not bounds is None:
        normed_train_x = (train_x - bounds[0])/(bounds[1]-bounds[0])
    else:
        normed_train_x = train_x 
    
    bounds = torch.tensor(np.array([np.zeros(train_x.shape[1]),np.ones(train_x.shape[1])]))

    # Create hyper parameter bounds
    noise_constraint = Interval(1e-6, 0.1)
    if use_ard:
        lengthscale_constraint = Interval(0.005, 20)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(normed_train_x.shape[1]))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=normed_train_x.device, dtype=train_y.dtype)
    ard_dims = normed_train_x.shape[1] if use_ard else None
    model = GP(
        train_x=normed_train_x,
        train_y=normed_train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
        mean_y = mean_y,
        std_y = std_y,
        bounds = bounds,
    ).to(device=normed_train_x.device, dtype=normed_train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.initialize(**hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = 0.5
        hypers["likelihood.noise"] = 0.00005
        model.initialize(**hypers)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(normed_train_x)
        loss = -mll(output, normed_train_y)
        loss.backward()
        optimizer.step()
    
    #print('loss = ',loss)

    # Switch to eval mode
    model.eval()
    likelihood.eval()
    t1 = time.time()
    #print('GP training time = ',t1-t0)
    return model
